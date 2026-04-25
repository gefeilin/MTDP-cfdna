from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .config import TARGET_SPECS


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


def build_waterfall_figure(explanation, *, top_n: int = 8) -> go.Figure:
    shap_series = explanation.shap_series.copy()
    if abs(explanation.other_baseline_residual) > 1e-8:
        shap_series.loc["Other baseline residual"] = explanation.other_baseline_residual

    top_feature_count = min(len(shap_series), max(1, int(top_n) - 1))
    selected = shap_series.abs().sort_values(ascending=False).head(top_feature_count)
    plot_series = shap_series.loc[selected.index].copy()
    remainder = shap_series.drop(plot_series.index, errors="ignore")
    if not remainder.empty:
        plot_series.loc["All other features"] = float(remainder.sum())

    plot_series = plot_series.sort_values()
    labels = []
    for feature_name in plot_series.index:
        feature_value = explanation.feature_values.get(feature_name, "")
        if feature_name in {"All other features", "Other baseline residual"}:
            labels.append(feature_name)
        else:
            labels.append(f"{feature_name} = {feature_value}")

    scale = 100.0 if explanation.target_key != "fev1_1y" else 1.0
    baseline = float(explanation.base_value) * scale
    deltas = plot_series.to_numpy(dtype=float) * scale
    prediction = float(explanation.prediction) * scale

    if explanation.target_key == "fev1_1y":
        unit_label = explanation.unit_label
    else:
        unit_label = TARGET_SPECS[explanation.target_key]["unit"]

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["absolute"] + ["relative"] * len(deltas) + ["total"],
            y=["Baseline"] + labels + ["Prediction"],
            x=[baseline] + deltas.tolist() + [0.0],
            connector={"line": {"color": "rgba(80,80,80,0.35)"}},
            decreasing={"marker": {"color": "#c62828"}},
            increasing={"marker": {"color": "#2e7d32"}},
            totals={"marker": {"color": "#1565c0"}},
        )
    )
    fig.update_layout(
        **_shared_layout(
            f"SHAP waterfall: {TARGET_SPECS[explanation.target_key]['label']}",
            height=520,
        ),
        xaxis_title=unit_label,
        yaxis_title="",
    )
    fig.add_annotation(
        x=prediction,
        y=-0.14,
        yref="paper",
        text=f"Prediction = {prediction:.2f}{'' if explanation.target_key == 'fev1_1y' else '%'}",
        showarrow=False,
        font=dict(size=12, color="#1565c0"),
    )
    fig.update_yaxes(automargin=True)
    return fig
