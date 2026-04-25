from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (
    FEATURE_LABEL_MAP,
    SHAP_BACKGROUND_SIZE,
    SHAP_L1_REG,
    SHAP_CACHE_DIR,
    SHAP_EXPLAINER_DIR,
    SHAP_NSAMPLES,
    TARGET_SPECS,
    USE_SAVED_EXPLAINER_DEFAULT,
    USE_SAVED_SHAP_DEFAULT,
)
from .metadata import format_feature_value_for_display


_EXPLAINER_CACHE: dict[tuple[str, str], object] = {}
logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("shap.explainers._kernel").setLevel(logging.WARNING)


@dataclass
class ExplanationResult:
    target_key: str
    prediction: float
    base_value: float
    shap_series: pd.Series
    other_baseline_residual: float
    feature_values: pd.Series
    from_cache: bool
    unit_label: str
    source: str


class PreparedTargetModel(torch.nn.Module):
    def __init__(self, service, target_key: str):
        super().__init__()
        self.service = service
        self.base_model = service.model
        self.target_key = target_key
        self.prepared_dim = int(service.cohort_combined_prepared.shape[1] // 2)
        self.horizon_bins = int(
            np.searchsorted(
                service.time_bin_edges_years,
                2.0,
                side="left",
            )
            + 1
        )
        self.register_buffer(
            "fev1_basis_vector",
            torch.tensor(service.fev1_basis_vector, dtype=torch.float32),
        )
        self.fev1_mean = float(service.fev1_scale.mean)
        self.fev1_std = float(service.fev1_scale.std)
        self.outcome_name_to_idx = {
            cfg["name"]: idx for idx, cfg in enumerate(service.outcome_configs)
        }

    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        x = combined_input[:, : self.prepared_dim]
        mask = combined_input[:, self.prepared_dim :]

        ae_out, _ = self.base_model.autoencoder(x, mask=mask)
        ae_out = self.base_model.linear_layer(ae_out).squeeze(-1)
        shared_out = self.base_model.shared_net(ae_out)
        base_h = torch.cat([ae_out, shared_out], dim=1)
        mask_augmented_h = torch.cat([base_h, mask], dim=1)

        out_list = [cs_net(mask_augmented_h) for cs_net in self.base_model.cs_nets]
        out = torch.cat(out_list, dim=1)
        out = self.base_model.output_layer(out)
        out = torch.softmax(out, dim=1)
        out = out.view(-1, self.base_model.num_Event, self.base_model.num_Category)

        outcome_preds = []
        for cfg, net in zip(self.base_model.outcome_configs, self.base_model.outcome_pred_nets):
            head_input = mask_augmented_h if cfg["task_type"] == "longitudinal_regression" else base_h
            outcome_preds.append(net(head_input))

        if self.target_key == "mortality_2y":
            return out[:, 0, : self.horizon_bins].sum(dim=1, keepdim=True)
        if self.target_key == "fev1_1y":
            pred_scaled = outcome_preds[self.outcome_name_to_idx["FEV1"]] @ self.fev1_basis_vector
            pred_unscaled = pred_scaled * self.fev1_std + self.fev1_mean
            return pred_unscaled.unsqueeze(1)
        if self.target_key == "severe_ACR":
            return outcome_preds[self.outcome_name_to_idx["severe_ACR"]].reshape(-1, 1)
        if self.target_key == "ever_clinical_AMR":
            return outcome_preds[self.outcome_name_to_idx["ever_clinical_AMR"]].reshape(-1, 1)
        if self.target_key == "BLAD":
            return outcome_preds[self.outcome_name_to_idx["BLAD"]].reshape(-1, 1)
        raise KeyError(f"Unsupported target_key: {self.target_key}")


class KernelShapPredictor:
    def __init__(self, service, target_key: str):
        self.target_key = str(target_key)
        self.service = None
        self.device = None
        self.wrapped_model = None
        self.rebind(service)

    def __getstate__(self):
        return {"target_key": self.target_key}

    def __setstate__(self, state):
        self.target_key = str(state["target_key"])
        self.service = None
        self.device = None
        self.wrapped_model = None

    def rebind(self, service) -> None:
        self.service = service
        self.device = service.device
        wrapped_model = PreparedTargetModel(service, self.target_key).to(service.device)
        wrapped_model.eval()
        self.wrapped_model = wrapped_model

    def __call__(self, z):
        if self.wrapped_model is None or self.device is None:
            raise RuntimeError(
                "Saved SHAP explainer predictor is not bound to a live prediction service."
            )
        tensor = torch.tensor(np.asarray(z, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.wrapped_model(tensor).detach().cpu().numpy().reshape(-1)


def _build_background_indices(service, bg_n: int, seed: int = 74) -> np.ndarray:
    cohort = service.schema.cohort_df
    strata_df = pd.DataFrame({"event": cohort["event"].astype(str)})
    category_columns = (
        service.schema.passthrough_category_columns + service.schema.embedding_category_columns
    )
    for column in category_columns:
        strata_df[column] = cohort[column].map(
            lambda value: "MISSING" if pd.isna(value) else str(value)
        )
    strata_keys = strata_df.astype(str).agg("|".join, axis=1)
    strata_frame = pd.DataFrame(
        {"row_index": np.arange(len(cohort), dtype=int), "stratum": strata_keys}
    )

    counts = strata_frame["stratum"].value_counts().sort_index()
    desired = counts / float(counts.sum()) * float(bg_n)
    allocation = np.floor(desired).astype(int).clip(lower=0, upper=counts)

    remaining = int(bg_n - allocation.sum())
    remainders = (desired - allocation).sort_values(ascending=False)
    for stratum in remainders.index:
        if remaining <= 0:
            break
        if allocation.loc[stratum] < counts.loc[stratum]:
            allocation.loc[stratum] += 1
            remaining -= 1

    rng = np.random.default_rng(int(seed))
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    for stratum, take_n in allocation.items():
        if int(take_n) <= 0:
            continue
        candidates = strata_frame.loc[
            strata_frame["stratum"] == stratum, "row_index"
        ].to_numpy(dtype=int)
        chosen = rng.choice(candidates, size=int(take_n), replace=False)
        selected_indices.extend(np.sort(chosen.astype(int)).tolist())
        selected_set.update(int(index) for index in chosen.tolist())

    if len(selected_indices) < bg_n:
        remaining_pool = np.asarray(
            [index for index in range(len(cohort)) if index not in selected_set],
            dtype=int,
        )
        extra_n = min(int(bg_n - len(selected_indices)), int(remaining_pool.size))
        if extra_n > 0:
            extra = rng.choice(remaining_pool, size=extra_n, replace=False)
            selected_indices.extend(np.sort(extra.astype(int)).tolist())

    return np.asarray(sorted(set(selected_indices)), dtype=int)


def _aggregate_single_shap(service, shap_values_prepared: np.ndarray) -> pd.Series:
    if shap_values_prepared.ndim == 3 and shap_values_prepared.shape[-1] == 1:
        shap_values_prepared = shap_values_prepared[..., 0]
    if shap_values_prepared.ndim != 2 or shap_values_prepared.shape[0] != 1:
        raise ValueError(f"Unexpected SHAP shape: {shap_values_prepared.shape}")

    aggregated = np.zeros(len(service.schema.feature_names), dtype=np.float32)
    for feature_idx, feature_name in enumerate(service.schema.feature_names):
        cols = service.feature_block_mapping[feature_name]["x_idx"]
        aggregated[feature_idx] = shap_values_prepared[0, cols].sum(axis=0)

    display_index = [FEATURE_LABEL_MAP.get(name, name) for name in service.schema.feature_names]
    return pd.Series(aggregated, index=display_index, dtype=np.float32)


def _make_cache_path(target_key: str, patient_index: int) -> Path:
    if target_key == "mortality_2y":
        return SHAP_CACHE_DIR / f"individual_patient{patient_index}_24m_bg64_stratified_event_categorical_v1_seed74_ns500.pkl"
    return SHAP_CACHE_DIR / f"individual_{target_key}_patient{patient_index}_bg64_ns500.pkl"


def _make_explainer_path(target_key: str, background_size: int) -> Path:
    return SHAP_EXPLAINER_DIR / f"kernel_explainer_{target_key}_bg{int(background_size)}.dill"


def _candidate_explainer_paths(target_key: str, background_size: int) -> list[Path]:
    exact_path = _make_explainer_path(target_key, background_size)
    candidates = [exact_path]
    pattern = f"kernel_explainer_{target_key}_bg*.dill"
    discovered = sorted(
        SHAP_EXPLAINER_DIR.glob(pattern),
        key=lambda path: (
            abs(
                int(path.stem.rsplit("_bg", 1)[-1]) - int(background_size)
            ) if "_bg" in path.stem else 10**9,
            path.name,
        ),
    )
    for path in discovered:
        if path not in candidates:
            candidates.append(path)
    return candidates


def _install_pickle_compat_aliases() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    if hasattr(np.core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def _read_pickle_compat(path: Path):
    _install_pickle_compat_aliases()
    return pd.read_pickle(path)


def _read_dill_compat(path: Path):
    try:
        import dill
    except ImportError:
        return None
    _install_pickle_compat_aliases()
    with path.open("rb") as handle:
        return dill.load(handle)


def _feature_value_series_for_display(detail: dict) -> pd.Series:
    raw_row = detail["raw_features"].iloc[0]
    return pd.Series(
        {
            FEATURE_LABEL_MAP.get(column, column): format_feature_value_for_display(
                FEATURE_LABEL_MAP.get(column, column),
                raw_row[column],
            )
            for column in raw_row.index
        }
    )


def _use_saved_shap_enabled() -> bool:
    return os.getenv("CFDNA_USE_SAVED_SHAP", "1" if USE_SAVED_SHAP_DEFAULT else "0") == "1"


def _use_saved_explainer_enabled() -> bool:
    return (
        os.getenv(
            "CFDNA_USE_SAVED_EXPLAINER",
            "1" if USE_SAVED_EXPLAINER_DEFAULT else "0",
        )
        == "1"
    )


def _maybe_load_cached_explanation(service, target_key: str, patient_index: int | None, detail: dict):
    if patient_index is None:
        return None
    if target_key == "fev1_1y" and service.fev1_scale.scaled_fallback:
        return None
    if not _use_saved_shap_enabled():
        return None

    cache_path = _make_cache_path(target_key, patient_index)
    if not cache_path.exists():
        return None

    try:
        cached = _read_pickle_compat(cache_path)
    except Exception:
        return None
    return ExplanationResult(
        target_key=target_key,
        prediction=float(cached["prediction"]),
        base_value=float(cached["base_value"]),
        shap_series=cached["shap_series"].copy(),
        other_baseline_residual=float(cached.get("other_baseline_residual", 0.0)),
        feature_values=_feature_value_series_for_display(detail),
        from_cache=True,
        unit_label=detail["fev1_display_label"] if target_key == "fev1_1y" else TARGET_SPECS[target_key]["unit"],
        source="saved_cache",
    )


def build_kernel_explainer(service, target_key: str, *, background_size: int = SHAP_BACKGROUND_SIZE):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("The `shap` package is required for SHAP explanations.") from exc

    predictor = KernelShapPredictor(service, target_key)
    bg_indices = _build_background_indices(
        service,
        bg_n=min(int(background_size), int(service.cohort_combined_prepared.shape[0])),
    )
    background = service.cohort_combined_prepared[bg_indices]
    return shap.KernelExplainer(predictor, background, link="identity")


def save_kernel_explainer(
    service,
    target_key: str,
    *,
    background_size: int = SHAP_BACKGROUND_SIZE,
    overwrite: bool = False,
) -> Path:
    try:
        import dill
    except ImportError as exc:
        raise RuntimeError("The `dill` package is required to save SHAP explainers.") from exc

    path = _make_explainer_path(target_key, background_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path

    explainer = build_kernel_explainer(service, target_key, background_size=background_size)
    with path.open("wb") as handle:
        dill.dump(explainer, handle)
    _EXPLAINER_CACHE[(str(path.resolve()), target_key)] = explainer
    return path


def _maybe_load_saved_explainer(service, target_key: str, background_size: int):
    if not _use_saved_explainer_enabled():
        return None
    for path in _candidate_explainer_paths(target_key, background_size):
        if not path.exists():
            continue

        cache_key = (str(path.resolve()), target_key)
        explainer = _EXPLAINER_CACHE.get(cache_key)
        if explainer is None:
            try:
                explainer = _read_dill_compat(path)
            except Exception:
                continue
            if explainer is None:
                continue
            _EXPLAINER_CACHE[cache_key] = explainer

        predictor = getattr(getattr(explainer, "model", None), "f", None)
        if not hasattr(predictor, "rebind"):
            continue

        try:
            predictor.rebind(service)
        except Exception:
            continue
        return explainer
    return None


def _normalize_shap_array(shap_raw) -> np.ndarray:
    shap_array = np.asarray(shap_raw[0] if isinstance(shap_raw, list) else shap_raw)
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)
    if shap_array.ndim == 3 and shap_array.shape[-1] == 1:
        shap_array = shap_array[..., 0]
    return shap_array


def _normalize_base_value(base_value) -> float:
    if isinstance(base_value, (list, np.ndarray)):
        return float(np.asarray(base_value).reshape(-1)[0])
    return float(base_value)


def _compute_from_explainer(
    service,
    detail: dict,
    target_key: str,
    explainer,
    *,
    nsamples: int,
    source: str,
) -> ExplanationResult:
    explain_np = np.asarray(detail["prepared_input"], dtype=np.float32)
    shap_raw = explainer.shap_values(
        explain_np,
        nsamples=int(nsamples),
        l1_reg=SHAP_L1_REG,
        silent=True,
        gc_collect=True,
    )
    shap_array = _normalize_shap_array(shap_raw)
    base_scalar = _normalize_base_value(explainer.expected_value)

    prediction_fn = getattr(getattr(explainer, "model", None), "f", None)
    if prediction_fn is None:
        raise RuntimeError("Explainer is missing its prediction function.")

    prediction = float(np.asarray(prediction_fn(explain_np)).reshape(-1)[0])
    shap_series = _aggregate_single_shap(service, shap_array)
    residual = prediction - base_scalar - float(shap_series.sum())

    return ExplanationResult(
        target_key=target_key,
        prediction=prediction,
        base_value=base_scalar,
        shap_series=shap_series,
        other_baseline_residual=float(residual),
        feature_values=_feature_value_series_for_display(detail),
        from_cache=(source == "saved_cache"),
        unit_label=detail["fev1_display_label"] if target_key == "fev1_1y" else TARGET_SPECS[target_key]["unit"],
        source=source,
    )


def compute_individual_explanation(
    service,
    detail: dict,
    target_key: str,
    *,
    patient_index: int | None = None,
    force_recompute: bool = False,
    background_size: int = SHAP_BACKGROUND_SIZE,
    nsamples: int = SHAP_NSAMPLES,
) -> ExplanationResult:
    _ = force_recompute

    cached = _maybe_load_cached_explanation(service, target_key, patient_index, detail)
    if cached is not None:
        return cached

    saved_explainer = _maybe_load_saved_explainer(service, target_key, background_size)
    if saved_explainer is not None:
        return _compute_from_explainer(
            service,
            detail,
            target_key,
            saved_explainer,
            nsamples=nsamples,
            source="saved_explainer",
        )

    raise RuntimeError(
        "No saved SHAP cache or saved explainer is available for this target. "
        "Rebuild the saved explainers offline before rendering SHAP."
    )
