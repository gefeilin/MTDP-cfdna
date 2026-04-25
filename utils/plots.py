from __future__ import annotations

import base64
from io import BytesIO
import importlib
import sys
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import CFDNA_ROOT_DIR, TARGET_SPECS


def format_target_value(target_key: str, value: float, *, fev1_scaled_fallback: bool = False) -> str:
    if target_key == "fev1_1y":
        return f"{value:.2f}" if fev1_scaled_fallback else f"{value:.1f}"
    return f"{100.0 * value:.1f}%"


def _shared_layout(title: str, *, height: int | None = None) -> dict:
    layout = {
        "template": "plotly_white",
        "title": {
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"b": 16},
        },
        "margin": {"l": 56, "r": 24, "t": 92, "b": 84},
        "legend": {"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0},
        "hovermode": "x unified",
    }
    if height is not None:
        layout["height"] = int(height)
    return layout


def build_survival_figure(original: dict, updated: dict | None = None) -> go.Figure:
    fig = go.Figure()
    x = np.asarray(original["time_years"], dtype=float)
    original_curve = np.asarray(
        original.get("survival_curve_mc_mean", original["survival_curve"]),
        dtype=float,
    )

    if "survival_curve_lower" in original and "survival_curve_upper" in original:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate(
                    [100.0 * original["survival_curve_upper"], 100.0 * original["survival_curve_lower"][::-1]]
                ),
                fill="toself",
                fillcolor="rgba(21,101,192,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name="Original MC interval",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=100.0 * np.asarray(original["survival_curve_upper"], dtype=float),
                mode="lines",
                line=dict(color="rgba(21,101,192,0.35)", width=1.5, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=100.0 * np.asarray(original["survival_curve_lower"], dtype=float),
                mode="lines",
                line=dict(color="rgba(21,101,192,0.35)", width=1.5, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=100.0 * original_curve,
            mode="lines+markers",
            name="Original cumulative risk",
            line=dict(color="#1565c0", width=3),
        )
    )

    if updated is not None:
        updated_curve = np.asarray(
            updated.get("survival_curve_mc_mean", updated["survival_curve"]),
            dtype=float,
        )
        if "survival_curve_lower" in updated and "survival_curve_upper" in updated:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate(
                        [100.0 * updated["survival_curve_upper"], 100.0 * updated["survival_curve_lower"][::-1]]
                    ),
                    fill="toself",
                    fillcolor="rgba(198,40,40,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="Updated MC interval",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=100.0 * np.asarray(updated["survival_curve_upper"], dtype=float),
                    mode="lines",
                    line=dict(color="rgba(198,40,40,0.35)", width=1.5, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=100.0 * np.asarray(updated["survival_curve_lower"], dtype=float),
                    mode="lines",
                    line=dict(color="rgba(198,40,40,0.35)", width=1.5, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=100.0 * updated_curve,
                mode="lines+markers",
                name="Updated cumulative risk",
                line=dict(color="#c62828", width=3, dash="dash"),
            )
        )

    fig.update_layout(
        **_shared_layout("Cumulative mortality risk"),
        xaxis_title="Years after transplant",
        yaxis_title="Risk (%)",
    )
    fig.update_yaxes(ticksuffix="%")
    return fig


def build_fev1_figure(original: dict, updated: dict | None = None) -> go.Figure:
    fig = go.Figure()
    x = np.asarray(original["months"], dtype=float)
    yaxis_title = original["fev1_display_label"]
    original_curve = np.asarray(
        original.get("fev1_curve_mc_mean", original["fev1_curve"]),
        dtype=float,
    )

    if "fev1_curve_lower" in original and "fev1_curve_upper" in original:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([original["fev1_curve_upper"], original["fev1_curve_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(0,121,107,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name="Original MC interval",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.asarray(original["fev1_curve_upper"], dtype=float),
                mode="lines",
                line=dict(color="rgba(0,121,107,0.35)", width=1.5, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.asarray(original["fev1_curve_lower"], dtype=float),
                mode="lines",
                line=dict(color="rgba(0,121,107,0.35)", width=1.5, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=original_curve,
            mode="lines",
            name="Original trajectory",
            line=dict(color="#00796b", width=3),
        )
    )

    if updated is not None:
        updated_curve = np.asarray(
            updated.get("fev1_curve_mc_mean", updated["fev1_curve"]),
            dtype=float,
        )
        if "fev1_curve_lower" in updated and "fev1_curve_upper" in updated:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([updated["fev1_curve_upper"], updated["fev1_curve_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(230,81,0,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="Updated MC interval",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.asarray(updated["fev1_curve_upper"], dtype=float),
                    mode="lines",
                    line=dict(color="rgba(230,81,0,0.35)", width=1.5, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.asarray(updated["fev1_curve_lower"], dtype=float),
                    mode="lines",
                    line=dict(color="rgba(230,81,0,0.35)", width=1.5, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=updated_curve,
                mode="lines",
                name="Updated trajectory",
                line=dict(color="#e65100", width=3, dash="dash"),
            )
        )

    fig.update_layout(
        **_shared_layout("Predicted FEV1 trajectory"),
        xaxis_title="Months after transplant",
        yaxis_title=yaxis_title,
    )
    return fig


def format_month_label(years_value: float) -> str:
    return f"{int(round(float(years_value) * 12))}m"


@lru_cache(maxsize=1)
def _load_notebook_shap_v2():
    package_parent = str(CFDNA_ROOT_DIR)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    existing = sys.modules.get("shap_v2")
    existing_file = getattr(existing, "__file__", "") if existing is not None else ""
    if existing is not None and (not existing_file or not str(existing_file).startswith(package_parent)):
        for module_name in list(sys.modules):
            if module_name == "shap_v2" or module_name.startswith("shap_v2."):
                sys.modules.pop(module_name, None)

    return importlib.import_module("shap_v2")


def _notebook_waterfall_target_spec(explanation, plot_shap_series: pd.Series) -> tuple[float, str, tuple[float, float]]:
    if explanation.target_key in {"mortality_2y", "severe_ACR", "ever_clinical_AMR", "BLAD"}:
        if explanation.target_key == "mortality_2y":
            return 100.0, f"Mortality risk {format_month_label(2.0)} (%)", (-30.0, 130.0)

        notebook_label_map = {
            "severe_ACR": "severe_ACR probability (%)",
            "ever_clinical_AMR": "ever_clinical_AMR probability (%)",
            "BLAD": "BLAD probability (%)",
        }
        return 100.0, notebook_label_map[explanation.target_key], (-30.0, 130.0)

    pred_value = float(explanation.prediction)
    base_value = float(explanation.base_value)
    lower = min(pred_value, base_value, float(plot_shap_series.sum() + base_value))
    upper = max(pred_value, base_value, float(plot_shap_series.sum() + base_value))
    pad = max(0.5, 0.2 * (upper - lower + 1e-6))
    return 1.0, "1-year FEV1 predicted value", (lower - pad, upper + pad)


def _waterfall_target_config(explanation, plot_shap_series: pd.Series) -> tuple[float, str, tuple[float, float]]:
    if explanation.target_key in {"mortality_2y", "severe_ACR", "ever_clinical_AMR", "BLAD"}:
        if explanation.target_key == "mortality_2y":
            target_label = f"Mortality risk {format_month_label(2.0)} (%)"
        else:
            target_label = f"{TARGET_SPECS[explanation.target_key]['label']} (%)"
        return 100.0, target_label, (-30.0, 130.0)

    pred_value = float(explanation.prediction)
    base_value = float(explanation.base_value)
    lower = min(pred_value, base_value, float(plot_shap_series.sum() + base_value))
    upper = max(pred_value, base_value, float(plot_shap_series.sum() + base_value))
    pad = max(0.5, 0.2 * (upper - lower + 1e-6))
    return 1.0, explanation.unit_label or TARGET_SPECS[explanation.target_key]["label"], (lower - pad, upper + pad)


def _apply_notebook_style_top_axes(
    figure,
    *,
    base_value: float,
    prediction: float,
    xlim: tuple[float, float],
    target_label: str,
    average_prefix: str,
    predicted_prefix: str,
    predicted_label: str | None,
) -> None:
    if len(figure.axes) < 3:
        return

    xmin, xmax = xlim
    fig = figure
    avg_axis = figure.axes[1]
    pred_axis = figure.axes[2]

    avg_axis.set_xlim(xmin, xmax)
    avg_axis.set_xticks([base_value, base_value + min(1e-8, max(abs(xmax), 1.0) * 1e-10)])
    avg_axis.set_xticklabels(
        [f"\n{average_prefix}\n{target_label}", f"\n= {base_value:0.03f}"],
        fontsize=12,
        ha="left",
    )

    pred_axis.set_xlim(xmin, xmax)
    pred_axis.set_xticks([prediction, prediction + min(1e-8, max(abs(xmax), 1.0) * 1e-10)])
    pred_header = predicted_label if predicted_label is not None else f"{predicted_prefix}\n{target_label}"
    pred_axis.set_xticklabels(
        [pred_header, f"= {prediction:0.03f}"],
        fontsize=12,
        ha="left",
    )

    tick_labels = pred_axis.xaxis.get_majorticklabels()
    if len(tick_labels) >= 2:
        tick_labels[0].set_transform(
            tick_labels[0].get_transform()
            + matplotlib.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans)
        )
        tick_labels[1].set_transform(
            tick_labels[1].get_transform()
            + matplotlib.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans)
        )
        tick_labels[1].set_color("#777777")

    avg_tick_labels = avg_axis.xaxis.get_majorticklabels()
    if len(avg_tick_labels) >= 2:
        avg_tick_labels[0].set_transform(
            avg_tick_labels[0].get_transform()
            + matplotlib.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans)
        )
        avg_tick_labels[1].set_transform(
            avg_tick_labels[1].get_transform()
            + matplotlib.transforms.ScaledTranslation(22 / 72.0, -1 / 72.0, fig.dpi_scale_trans)
        )
        avg_tick_labels[1].set_color("#777777")


def build_waterfall_image_data_url(
    explanation,
    *,
    top_n: int = 8,
    figsize: tuple[float, float] = (10.2, 3.25),
    dpi: int = 220,
    predicted_prefix: str = "Predicted",
    predicted_label: str | None = None,
) -> str:
    shap_series = explanation.shap_series.copy()
    feature_values = explanation.feature_values.copy()
    residual_feature_name = "__missing_residual_hidden__"

    top_feature_count = min(len(shap_series), max(1, int(top_n) - 1))
    top_feature_names = (
        shap_series.abs().sort_values(ascending=False).head(top_feature_count).index.tolist()
    )

    plot_shap_series = shap_series.copy()
    plot_feature_values = feature_values.copy()
    if abs(float(explanation.other_baseline_residual)) > 0:
        plot_shap_series.loc[residual_feature_name] = float(explanation.other_baseline_residual)
        plot_feature_values.loc[residual_feature_name] = ""

    remaining_feature_names = [
        name for name in plot_shap_series.index.tolist() if name not in top_feature_names
    ]
    ordered_feature_names = top_feature_names + remaining_feature_names
    order = np.array(
        [plot_shap_series.index.get_loc(name) for name in ordered_feature_names],
        dtype=int,
    )

    feature_display_values = pd.Series(
        [
            str(value) if not pd.isna(value) else ""
            for value in plot_feature_values.tolist()
        ],
        index=plot_feature_values.index,
        dtype=object,
    )
    if residual_feature_name in feature_display_values.index:
        feature_display_values.loc[residual_feature_name] = ""

    scale, target_label, xlim = _notebook_waterfall_target_spec(explanation, plot_shap_series)
    shap_v2 = _load_notebook_shap_v2()
    shap_explanation = shap_v2.Explanation(
        values=plot_shap_series.to_numpy(dtype=float) * scale,
        base_values=float(explanation.base_value) * scale,
        data=feature_display_values.to_numpy(dtype=object),
        display_data=feature_display_values.to_numpy(dtype=object),
        feature_names=plot_shap_series.index.tolist(),
    )

    plt.figure(figsize=figsize)
    custom_predicted_label = (
        predicted_label
        if predicted_label is not None
        else (predicted_prefix if predicted_prefix != "Predicted" else None)
    )
    shap_v2.plots.waterfall_v2(
        shap_explanation,
        max_display=int(top_n),
        show=False,
        xlim=xlim,
        order=order,
        target_label=target_label,
        average_prefix="Average predicted",
        predicted_prefix=predicted_prefix,
        predicted_label=custom_predicted_label,
    )
    figure = plt.gcf()
    figure.set_size_inches(*figsize)

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
