from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import BSpline

from .config import (
    CAT_EMBEDDING_DIM,
    CHECKPOINT_FALLBACK,
    FEATURE_LABEL_MAP,
    FEATURE_LABEL_REVERSE_MAP,
    FEV1_TARGET_MONTH,
    MC_INTERVAL_ALPHA,
    MC_RANDOM_SEED_DEFAULT,
    MC_SAMPLES_DEFAULT,
    MODEL_METADATA_PATH,
    MORTALITY_TARGET_YEARS,
    SHAP_CACHE_DIR,
    TARGET_SPECS,
)
from .metadata import build_display_frame, build_schema_artifacts, canonicalize_category_value
from .runtime import add_project_paths, install_runtime_shims


@dataclass(frozen=True)
class TrialSettings:
    checkpoint_path: Path
    network_settings: dict[str, Any]
    bspline_degree: int
    bspline_internal_knots: list[float]
    bspline_time_min: float
    bspline_time_max: float


@dataclass(frozen=True)
class FEV1Scale:
    mean: float
    std: float
    source: str
    scaled_fallback: bool = False


@dataclass
class PreparedBatch:
    raw_features: pd.DataFrame
    display_features: pd.DataFrame
    x: np.ndarray
    mask: np.ndarray
    subject_numbers: list[str]
    warnings: list[str]


def _format_horizon_suffix(years: float) -> str:
    return f"{float(years):.2f}".rstrip("0").rstrip(".")


@lru_cache(maxsize=1)
def _load_model_metadata() -> dict[str, Any]:
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata file is missing: {MODEL_METADATA_PATH}"
        )
    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_trial_settings() -> TrialSettings:
    metadata = _load_model_metadata()
    model_settings = metadata.get("model_settings", {})

    checkpoint_path = CHECKPOINT_FALLBACK
    checkpoint_relative_path = model_settings.get("checkpoint_relative_path")
    if checkpoint_relative_path:
        checkpoint_path = MODEL_METADATA_PATH.parents[1] / checkpoint_relative_path

    network_defaults = {
        "h_dim_shared": 88,
        "h_dim_CS": 80,
        "num_layers_shared": 1,
        "num_layers_CS": 2,
        "active_fn": "tanh",
        "keep_prob": 0.6925135489809056,
        "ae_out_dim": 64,
        "ae_hidden_dim1": 128,
        "ae_hidden_dim2": 64,
        "ae_num_heads": 8,
        "ae_num_layers": 1,
    }
    network_payload = model_settings.get("network_settings", {})
    network_settings = {
        key: network_payload.get(key, default)
        for key, default in network_defaults.items()
    }

    return TrialSettings(
        checkpoint_path=Path(checkpoint_path),
        network_settings={
            "h_dim_shared": int(network_settings["h_dim_shared"]),
            "h_dim_CS": int(network_settings["h_dim_CS"]),
            "num_layers_shared": int(network_settings["num_layers_shared"]),
            "num_layers_CS": int(network_settings["num_layers_CS"]),
            "active_fn": str(network_settings["active_fn"]),
            "keep_prob": float(network_settings["keep_prob"]),
            "ae_out_dim": int(network_settings["ae_out_dim"]),
            "ae_hidden_dim1": int(network_settings["ae_hidden_dim1"]),
            "ae_hidden_dim2": int(network_settings["ae_hidden_dim2"]),
            "ae_num_heads": int(network_settings["ae_num_heads"]),
            "ae_num_layers": int(network_settings["ae_num_layers"]),
        },
        bspline_degree=int(model_settings.get("bspline_degree", 3)),
        bspline_internal_knots=list(model_settings.get("bspline_internal_knots", [3.0, 9.0])),
        bspline_time_min=float(model_settings.get("bspline_time_min", 0.0)),
        bspline_time_max=float(model_settings.get("bspline_time_max", 15.0)),
    )


def _build_bspline_knots(
    degree: int,
    time_min: float,
    time_max: float,
    internal_knots: list[float],
) -> np.ndarray:
    return np.concatenate(
        (
            np.repeat(time_min, degree + 1),
            np.asarray(internal_knots, dtype=np.float64),
            np.repeat(time_max, degree + 1),
        )
    )


def _num_basis_from_settings(settings: TrialSettings) -> int:
    knots = _build_bspline_knots(
        degree=settings.bspline_degree,
        time_min=settings.bspline_time_min,
        time_max=settings.bspline_time_max,
        internal_knots=settings.bspline_internal_knots,
    )
    return int(len(knots) - settings.bspline_degree - 1)


def _build_outcome_configs(num_basis: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "FEV1",
            "task_type": "longitudinal_regression",
            "output_dim": int(num_basis),
            "output_activation": None,
            "outcome_idx": 0,
        },
        {
            "name": "severe_ACR",
            "task_type": "binary_classification",
            "output_dim": 1,
            "output_activation": "sigmoid",
            "outcome_idx": 1,
        },
        {
            "name": "ever_clinical_AMR",
            "task_type": "binary_classification",
            "output_dim": 1,
            "output_activation": "sigmoid",
            "outcome_idx": 2,
        },
        {
            "name": "BLAD",
            "task_type": "binary_classification",
            "output_dim": 1,
            "output_activation": "sigmoid",
            "outcome_idx": 3,
        },
    ]


class CfDNAPredictionService:
    def __init__(self) -> None:
        install_runtime_shims()
        add_project_paths()

        from engines.multitask_deephit_v1_4_cfdna_torch_death_only_maskhead_v5_nosite_nopeakdsa_noweights_noaucddpct import (
            ModelDeepHit_Multitask,
            TIME_BIN_EDGES_YEARS,
            TIME_BIN_REPRESENTATIVES_YEARS,
            prepare_input_dims,
        )

        self.ModelDeepHit_Multitask = ModelDeepHit_Multitask
        self.prepare_input_dims = prepare_input_dims
        self.time_bin_edges_years = np.asarray(TIME_BIN_EDGES_YEARS, dtype=np.float32)
        self.display_survival_bins = int(len(self.time_bin_edges_years))
        self.num_category = int(len(TIME_BIN_REPRESENTATIVES_YEARS))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.schema = build_schema_artifacts()
        self.trial_settings = _load_trial_settings()
        self.num_basis = _num_basis_from_settings(self.trial_settings)
        self.outcome_configs = _build_outcome_configs(self.num_basis)
        self.display_feature_names = [
            FEATURE_LABEL_MAP.get(name, name) for name in self.schema.feature_names
        ]

        cohort_batch = self.prepare_batch(
            self.schema.cohort_df[self.schema.feature_names].copy(),
            subject_numbers=self.schema.cohort_df["SUBJECT_NUMBER"].astype(str).tolist(),
        )
        self.cohort_batch = cohort_batch

        self.model = self._build_model(cohort_batch.x)
        self.cohort_combined_prepared = self._prepare_combined(
            cohort_batch.x, cohort_batch.mask
        )
        self.feature_block_mapping = self._build_feature_block_mapping()

        self.fev1_month_grid = np.linspace(0.0, 12.0, 61, dtype=np.float32)
        self.fev1_basis_vector = self.build_bspline_basis_vector(FEV1_TARGET_MONTH)
        self.fev1_basis_matrix = self.build_bspline_basis_matrix(self.fev1_month_grid)
        self.fev1_scale = self._resolve_fev1_scale()
        self.cohort_display_features = cohort_batch.display_features.copy()

    def _build_model(self, reference_x: np.ndarray) -> torch.nn.Module:
        input_dims = self.prepare_input_dims(
            X=reference_x,
            num_event=1,
            num_category=self.num_category,
            num_numerical=self.schema.num_numerical,
            cat_cardinalities=self.schema.cat_cardinalities,
            cat_embedding_dim=CAT_EMBEDDING_DIM,
        )
        model = self.ModelDeepHit_Multitask(
            input_dims=input_dims,
            network_settings=self.trial_settings.network_settings,
            outcome_configs=self.outcome_configs,
            writer=None,
        ).to(self.device)
        try:
            state_dict = torch.load(
                self.trial_settings.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(
                self.trial_settings.checkpoint_path,
                map_location=self.device,
            )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def build_bspline_basis_vector(self, month_value: float) -> np.ndarray:
        knots = _build_bspline_knots(
            degree=self.trial_settings.bspline_degree,
            time_min=self.trial_settings.bspline_time_min,
            time_max=self.trial_settings.bspline_time_max,
            internal_knots=self.trial_settings.bspline_internal_knots,
        )
        coeff = np.eye(self.num_basis, dtype=np.float64)
        basis_functions = [
            BSpline(knots, coeff[index], self.trial_settings.bspline_degree, extrapolate=False)
            for index in range(self.num_basis)
        ]
        return np.asarray(
            [basis(month_value) for basis in basis_functions],
            dtype=np.float32,
        )

    def build_bspline_basis_matrix(self, month_grid: np.ndarray) -> np.ndarray:
        return np.stack(
            [self.build_bspline_basis_vector(float(month)) for month in month_grid],
            axis=0,
        )

    def _resolve_fev1_scale(self) -> FEV1Scale:
        env_mean = os.getenv("CFDNA_FEV1_MEAN")
        env_std = os.getenv("CFDNA_FEV1_STD")
        if env_mean is not None and env_std is not None:
            return FEV1Scale(
                mean=float(env_mean),
                std=float(env_std),
                source="environment",
                scaled_fallback=False,
            )

        metadata_scale = self._load_fev1_scale_metadata()
        if metadata_scale is not None:
            return metadata_scale

        if os.getenv("CFDNA_ENABLE_PICKLE_SCALE_INFERENCE", "0") == "1":
            inferred = self._infer_fev1_scale_from_saved_cache()
            if inferred is not None:
                return inferred

        return FEV1Scale(
            mean=0.0,
            std=1.0,
            source="scaled-fallback",
            scaled_fallback=True,
        )

    def _load_fev1_scale_metadata(self) -> FEV1Scale | None:
        metadata = _load_model_metadata()
        scale_payload = metadata.get("fev1_scale", {})
        mean = scale_payload.get("mean")
        std = scale_payload.get("std")
        if mean is None or std is None:
            return None

        return FEV1Scale(
            mean=float(mean),
            std=float(std),
            source=f"metadata:{MODEL_METADATA_PATH.name}",
            scaled_fallback=False,
        )

    def _infer_fev1_scale_from_saved_cache(self) -> FEV1Scale | None:
        candidate_paths = sorted(
            SHAP_CACHE_DIR.glob("individual_fev1_1y_patient*_bg64_ns500.pkl")
        )
        if len(candidate_paths) < 2:
            return None

        reference_scaled: list[float] = []
        cached_unscaled: list[float] = []

        for cache_path in candidate_paths:
            try:
                patient_token = cache_path.stem.split("_patient", 1)[1].split("_", 1)[0]
                patient_index = int(patient_token)
            except (IndexError, ValueError):
                continue

            try:
                cached = pd.read_pickle(cache_path)
            except Exception:
                continue

            prediction = cached.get("prediction")
            if prediction is None:
                continue

            row_batch = PreparedBatch(
                raw_features=self.cohort_batch.raw_features.iloc[[patient_index]].copy(),
                display_features=self.cohort_batch.display_features.iloc[[patient_index]].copy(),
                x=self.cohort_batch.x[patient_index : patient_index + 1].copy(),
                mask=self.cohort_batch.mask[patient_index : patient_index + 1].copy(),
                subject_numbers=[self.cohort_batch.subject_numbers[patient_index]],
                warnings=[],
            )
            predicted = self.predict_single(row_batch, mc_samples=0)
            reference_scaled.append(float(predicted["fev1_1y_scaled"]))
            cached_unscaled.append(float(prediction))

        if len(reference_scaled) < 2:
            return None

        x = np.asarray(reference_scaled, dtype=np.float64)
        y = np.asarray(cached_unscaled, dtype=np.float64)
        slope, intercept = np.polyfit(x, y, deg=1)
        if not np.isfinite(slope) or slope <= 0:
            return None

        return FEV1Scale(
            mean=float(intercept),
            std=float(slope),
            source="saved-shap-cache",
            scaled_fallback=False,
        )

    def prepare_batch(
        self,
        raw_df: pd.DataFrame,
        *,
        subject_numbers: list[str] | None = None,
    ) -> PreparedBatch:
        missing_columns = [
            column for column in self.schema.feature_names if column not in raw_df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Uploaded CSV is missing required predictors: "
                + ", ".join(missing_columns[:10])
            )

        warnings: list[str] = []
        ordered = raw_df[self.schema.feature_names].copy()

        numerical_df = ordered[self.schema.numerical_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        numerical_mask = numerical_df.notna().to_numpy(dtype=np.float32)
        numerical_scaled = (
            (numerical_df - self.schema.numerical_mean) / self.schema.numerical_std
        ).fillna(0.0)

        passthrough_df = ordered[self.schema.passthrough_category_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        passthrough_mask = passthrough_df.notna().to_numpy(dtype=np.float32)
        passthrough_values = passthrough_df.fillna(0.0)

        embedding_arrays: list[np.ndarray] = []
        embedding_masks: list[np.ndarray] = []
        for column in self.schema.embedding_category_columns:
            raw_series = ordered[column]
            mask = raw_series.notna().to_numpy(dtype=np.float32)
            encoder = self.schema.embedding_codebooks[column]
            canonical_series = raw_series.map(canonicalize_category_value)
            encoded = canonical_series.map(lambda value: encoder.get(value, 0)).fillna(0).astype("float32")
            unknown_count = int(
                ((raw_series.notna()) & (~canonical_series.isin(list(encoder.keys())))).sum()
            )
            if unknown_count > 0:
                warnings.append(
                    f"{column}: {unknown_count} unseen category value(s) mapped to missing."
                )
            embedding_arrays.append(encoded.to_numpy(dtype=np.float32))
            embedding_masks.append(mask)

        x_parts = [
            numerical_scaled.to_numpy(dtype=np.float32),
            passthrough_values.to_numpy(dtype=np.float32),
        ]
        if embedding_arrays:
            x_parts.append(np.stack(embedding_arrays, axis=1).astype(np.float32))

        mask_parts = [
            numerical_mask.astype(np.float32),
            passthrough_mask.astype(np.float32),
        ]
        if embedding_masks:
            mask_parts.append(np.stack(embedding_masks, axis=1).astype(np.float32))

        subject_values = (
            subject_numbers
            if subject_numbers is not None
            else [
                str(value)
                for value in raw_df.get(
                    "SUBJECT_NUMBER",
                    pd.Series([f"uploaded_{index + 1}" for index in range(len(raw_df))]),
                ).tolist()
            ]
        )

        display_features = build_display_frame(ordered)
        return PreparedBatch(
            raw_features=ordered,
            display_features=display_features,
            x=np.concatenate(x_parts, axis=1).astype(np.float32),
            mask=np.concatenate(mask_parts, axis=1).astype(np.float32),
            subject_numbers=subject_values,
            warnings=warnings,
        )

    def _prepare_combined(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prepared_x, prepared_mask = self.model._prepare_features(x_tensor, mask_tensor)
        combined = torch.cat([prepared_x, prepared_mask], dim=1)
        return combined.detach().cpu().numpy()

    def _build_feature_block_mapping(self) -> dict[str, dict[str, list[int]]]:
        x_tensor = torch.tensor(
            self.cohort_batch.x[:1], dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.tensor(
            self.cohort_batch.mask[:1], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            prepared_x, _prepared_mask = self.model._prepare_features(x_tensor, mask_tensor)
        prepared_dim = int(prepared_x.shape[1])

        mapping: dict[str, dict[str, list[int]]] = {}
        cursor = 0
        for feature_index, feature_name in enumerate(self.schema.feature_names):
            if feature_index < self.schema.num_numerical:
                x_idx = [cursor]
                mask_idx = [prepared_dim + cursor]
                cursor += 1
            else:
                x_idx = list(range(cursor, cursor + CAT_EMBEDDING_DIM))
                mask_idx = list(
                    range(prepared_dim + cursor, prepared_dim + cursor + CAT_EMBEDDING_DIM)
                )
                cursor += CAT_EMBEDDING_DIM
            mapping[feature_name] = {
                "x_idx": x_idx,
                "mask_idx": mask_idx,
                "all_idx": x_idx + mask_idx,
            }
        return mapping

    def _predict_arrays(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        *,
        samples: int,
        stochastic: bool,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)

        original_mode = self.model.training
        self.model.train(stochastic)
        rng_seed = int(seed) if stochastic and seed is not None else None
        rng_devices = []
        if rng_seed is not None and self.device.type == "cuda":
            rng_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]

        draws: list[dict[str, np.ndarray]] = []
        horizon_bins = int(
            np.searchsorted(
                self.time_bin_edges_years,
                MORTALITY_TARGET_YEARS,
                side="left",
            )
            + 1
        )

        try:
            with torch.random.fork_rng(devices=rng_devices, enabled=rng_seed is not None):
                if rng_seed is not None:
                    torch.manual_seed(rng_seed)
                    if self.device.type == "cuda":
                        torch.cuda.manual_seed_all(rng_seed)

                with torch.no_grad():
                    for _ in range(max(samples, 1)):
                        survival_out, outcome_preds, _ = self.model(x_tensor, mask_tensor)
                        death_mass = survival_out[:, 0, :].detach().cpu().numpy()
                        cumulative_risk = np.cumsum(death_mass, axis=1)
                        survival_probability = np.clip(1.0 - cumulative_risk, 0.0, 1.0)
                        cumulative_risk_display = cumulative_risk[:, : self.display_survival_bins]
                        survival_probability_display = survival_probability[:, : self.display_survival_bins]

                        fev1_coefficients = outcome_preds[0].detach().cpu().numpy()
                        fev1_1y_scaled = fev1_coefficients @ self.fev1_basis_vector
                        fev1_curve_scaled = fev1_coefficients @ self.fev1_basis_matrix.T

                        severe_acr = outcome_preds[1].detach().cpu().numpy().reshape(-1)
                        clinical_amr = outcome_preds[2].detach().cpu().numpy().reshape(-1)
                        blad = outcome_preds[3].detach().cpu().numpy().reshape(-1)

                        draws.append(
                            {
                                "death_mass": death_mass,
                                "cumulative_risk": cumulative_risk,
                                "cumulative_risk_display": cumulative_risk_display,
                                "survival_probability": survival_probability,
                                "survival_probability_display": survival_probability_display,
                                "mortality_2y": cumulative_risk[:, horizon_bins - 1],
                                "fev1_1y_scaled": fev1_1y_scaled,
                                "fev1_1y": fev1_1y_scaled * self.fev1_scale.std + self.fev1_scale.mean,
                                "fev1_curve": fev1_curve_scaled * self.fev1_scale.std + self.fev1_scale.mean,
                                "severe_ACR": severe_acr,
                                "ever_clinical_AMR": clinical_amr,
                                "BLAD": blad,
                            }
                        )
        finally:
            self.model.train(original_mode)
            self.model.eval()

        keys = draws[0].keys()
        return {
            key: np.stack([draw[key] for draw in draws], axis=0)
            for key in keys
        }

    def predict_summary(self, batch: PreparedBatch) -> pd.DataFrame:
        predictions = self._predict_arrays(batch.x, batch.mask, samples=1, stochastic=False)
        summary_df = pd.DataFrame(
            {
                "subject_number": batch.subject_numbers,
                "mortality_2y": predictions["mortality_2y"][0],
                "fev1_1y": predictions["fev1_1y"][0],
                "severe_ACR": predictions["severe_ACR"][0],
                "ever_clinical_AMR": predictions["ever_clinical_AMR"][0],
                "BLAD": predictions["BLAD"][0],
            }
        )

        cumulative_risk = predictions["cumulative_risk_display"][0]
        for horizon_index, horizon_years in enumerate(
            self.time_bin_edges_years[: cumulative_risk.shape[1]]
        ):
            summary_df[f"mortality_risk_{_format_horizon_suffix(horizon_years)}y"] = cumulative_risk[
                :, horizon_index
            ]

        return summary_df

    def predict_single(
        self,
        batch: PreparedBatch,
        *,
        mc_samples: int = MC_SAMPLES_DEFAULT,
        mc_seed: int | None = MC_RANDOM_SEED_DEFAULT,
    ) -> dict[str, Any]:
        deterministic = self._predict_arrays(batch.x, batch.mask, samples=1, stochastic=False)
        result: dict[str, Any] = {
            "subject_number": batch.subject_numbers[0],
            "warnings": list(batch.warnings),
            "fev1_scaled_fallback": self.fev1_scale.scaled_fallback,
            "fev1_scale_source": self.fev1_scale.source,
            "fev1_display_label": (
                "1-year FEV1 (standardized)"
                if self.fev1_scale.scaled_fallback
                else TARGET_SPECS["fev1_1y"]["label"]
            ),
            "time_years": self.time_bin_edges_years[: deterministic["cumulative_risk_display"].shape[-1]],
            "months": self.fev1_month_grid,
            "prepared_input": self._prepare_combined(batch.x, batch.mask),
            "raw_features": batch.raw_features.copy(),
            "display_features": batch.display_features.copy(),
            "mortality_2y": float(deterministic["mortality_2y"][0, 0]),
            "fev1_1y": float(deterministic["fev1_1y"][0, 0]),
            "fev1_1y_scaled": float(deterministic["fev1_1y_scaled"][0, 0]),
            "severe_ACR": float(deterministic["severe_ACR"][0, 0]),
            "ever_clinical_AMR": float(deterministic["ever_clinical_AMR"][0, 0]),
            "BLAD": float(deterministic["BLAD"][0, 0]),
            "survival_curve": deterministic["cumulative_risk_display"][0, 0],
            "survival_probability": deterministic["survival_probability_display"][0, 0],
            "fev1_curve": deterministic["fev1_curve"][0, 0],
        }

        if mc_samples and mc_samples > 1:
            draws = self._predict_arrays(
                batch.x,
                batch.mask,
                samples=mc_samples,
                stochastic=True,
                seed=mc_seed,
            )
            lower_q = MC_INTERVAL_ALPHA / 2.0
            upper_q = 1.0 - lower_q

            for target_name in [
                "mortality_2y",
                "fev1_1y",
                "severe_ACR",
                "ever_clinical_AMR",
                "BLAD",
            ]:
                result[f"{target_name}_lower"] = float(
                    np.quantile(draws[target_name][:, 0], lower_q)
                )
                result[f"{target_name}_upper"] = float(
                    np.quantile(draws[target_name][:, 0], upper_q)
                )

            result["survival_curve_mc_mean"] = draws["cumulative_risk_display"][:, 0, :].mean(axis=0)
            survival_curve_lower = np.quantile(
                draws["cumulative_risk_display"][:, 0, :], lower_q, axis=0
            )
            survival_curve_upper = np.quantile(
                draws["cumulative_risk_display"][:, 0, :], upper_q, axis=0
            )
            survival_curve_lower = np.maximum.accumulate(
                np.clip(survival_curve_lower, 0.0, 1.0)
            )
            survival_curve_upper = np.maximum.accumulate(
                np.clip(survival_curve_upper, survival_curve_lower, 1.0)
            )
            result["survival_curve_lower"] = survival_curve_lower
            result["survival_curve_upper"] = survival_curve_upper
            result["fev1_curve_mc_mean"] = draws["fev1_curve"][:, 0, :].mean(axis=0)
            result["fev1_curve_lower"] = np.quantile(
                draws["fev1_curve"][:, 0, :], lower_q, axis=0
            )
            result["fev1_curve_upper"] = np.quantile(
                draws["fev1_curve"][:, 0, :], upper_q, axis=0
            )
        return result

    def lookup_known_patient_index(self, subject_number: str) -> int | None:
        return self.schema.subject_to_index.get(str(subject_number))


@lru_cache(maxsize=1)
def get_prediction_service() -> CfDNAPredictionService:
    return CfDNAPredictionService()
