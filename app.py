from __future__ import annotations

import base64
import io
import os
import uuid
from collections import OrderedDict

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import ALL, Input, Output, State, dcc, html, dash_table, no_update
from flask import Response

from utils.config import FEATURE_LABEL_MAP, MC_SAMPLES_DEFAULT, TARGET_SPECS
from utils.metadata import (
    build_category_dropdown_options,
    build_schema_artifacts,
    coerce_editor_record_for_model,
    load_baseline_frame,
    normalize_editor_record,
)
from utils.modeling import get_prediction_service
from utils.plots import (
    build_fev1_figure,
    build_survival_figure,
    build_waterfall_image_data_url,
    build_waterfall_png_bytes,
    format_target_value,
)
from utils.shap_utils import compute_individual_explanation


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "cfDNA Multitask DeepHit"

_SHAP_IMAGE_CACHE: OrderedDict[str, bytes] = OrderedDict()
_MAX_SHAP_IMAGE_CACHE_ITEMS = 32


def _store_shap_image(png_bytes: bytes) -> str:
    token = uuid.uuid4().hex
    _SHAP_IMAGE_CACHE[token] = png_bytes
    while len(_SHAP_IMAGE_CACHE) > _MAX_SHAP_IMAGE_CACHE_ITEMS:
        _SHAP_IMAGE_CACHE.popitem(last=False)
    return token


@app.server.route("/shap-image/<token>.png")
def serve_shap_image(token: str):
    png_bytes = _SHAP_IMAGE_CACHE.get(token)
    if png_bytes is None:
        return Response(status=404)
    return Response(png_bytes, mimetype="image/png")

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root {
        --ink: #132238;
        --muted: #5a6b7f;
        --sea: #0f766e;
        --sea-soft: #dff4f1;
        --fire: #c2410c;
        --fire-soft: #fff0e7;
        --sky: #1d4ed8;
        --sky-soft: #e8f0ff;
        --paper: #f6f8fb;
      }
      body {
        background: radial-gradient(circle at top right, #eef7ff 0%, #f6f8fb 38%, #f7f8fa 100%);
        color: var(--ink);
        font-family: "Segoe UI", Arial, sans-serif;
      }
      .hero {
        background:
          linear-gradient(135deg, rgba(13, 70, 99, 0.98), rgba(15, 118, 110, 0.92)),
          linear-gradient(45deg, rgba(255,255,255,0.08), rgba(255,255,255,0));
        border-radius: 0 0 22px 22px;
        box-shadow: 0 18px 40px rgba(19, 34, 56, 0.16);
        padding: 28px 32px;
        margin-bottom: 24px;
      }
      .hero-title {
        color: white;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-bottom: 6px;
      }
      .hero-sub {
        color: rgba(255,255,255,0.80);
        font-size: 0.92rem;
        margin-bottom: 0;
      }
      .soft-card {
        border: 0;
        border-radius: 18px;
        box-shadow: 0 12px 28px rgba(19, 34, 56, 0.08);
      }
      .metric-card {
        border: 0;
        border-radius: 16px;
        box-shadow: 0 10px 24px rgba(19, 34, 56, 0.08);
        min-height: 136px;
      }
      .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .metric-value {
        color: var(--ink);
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.1;
      }
      .metric-range {
        color: var(--muted);
        font-size: 0.86rem;
      }
      .upload-zone {
        width: 100%;
        min-height: 86px;
        border: 2px dashed #80b8ff;
        border-radius: 16px;
        background: rgba(255,255,255,0.75);
        color: var(--sky);
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 12px;
        transition: 0.2s ease;
        cursor: pointer;
      }
      .upload-zone:hover {
        background: #eef5ff;
        border-color: #3b82f6;
      }
      .section-title {
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: var(--ink);
      }
      .table-chip {
        font-size: 0.76rem;
        color: white;
        background: linear-gradient(135deg, #1d4ed8, #0f766e);
        border-radius: 999px;
        padding: 3px 10px;
      }
      @media (max-width: 768px) {
        .hero { padding: 22px 18px; }
        .hero-title { font-size: 1.42rem; }
        .metric-value { font-size: 1.45rem; }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""


def metric_card(title: str, target_key: str, detail: dict, updated: dict | None = None):
    fallback = bool(detail.get("fev1_scaled_fallback", False))
    original_value = format_target_value(target_key, float(detail[target_key]), fev1_scaled_fallback=fallback)
    lower = detail.get(f"{target_key}_lower")
    upper = detail.get(f"{target_key}_upper")
    range_text = ""
    if lower is not None and upper is not None:
        if target_key == "fev1_1y":
            range_text = f"MC interval: {lower:.2f} to {upper:.2f}"
        else:
            range_text = f"MC interval: {100*lower:.1f}% to {100*upper:.1f}%"

    updated_block = []
    if updated is not None:
        updated_value = format_target_value(
            target_key,
            float(updated[target_key]),
            fev1_scaled_fallback=bool(updated.get("fev1_scaled_fallback", False)),
        )
        updated_block = [
            html.Div("Updated", className="metric-label", style={"marginTop": "10px"}),
            html.Div(updated_value, className="metric-value", style={"fontSize": "1.2rem", "color": "#c2410c"}),
        ]

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="metric-label"),
                html.Div(original_value, className="metric-value"),
                html.Div(range_text, className="metric-range"),
                *updated_block,
            ]
        ),
        className="metric-card",
    )


def _parse_mortality_horizon_years(column_name: str) -> float:
    return float(column_name.removeprefix("mortality_risk_").removesuffix("y"))


def _format_year_label(years: float) -> str:
    return f"{float(years):.2f}".rstrip("0").rstrip(".")


def build_prediction_frames(prediction_records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame(prediction_records).copy()
    if raw_df.empty:
        return raw_df, raw_df

    display_df = pd.DataFrame({"Subject": raw_df["subject_number"].astype(str)})
    download_df = pd.DataFrame({"Subject": raw_df["subject_number"].astype(str)})

    mortality_columns = sorted(
        [column for column in raw_df.columns if column.startswith("mortality_risk_")],
        key=_parse_mortality_horizon_years,
    )
    for column in mortality_columns:
        year_label = _format_year_label(_parse_mortality_horizon_years(column))
        display_name = f"Mortality @ {year_label}y (%)"
        values_pct = 100.0 * raw_df[column].astype(float)
        display_df[display_name] = values_pct.map(lambda value: f"{value:.2f}")
        download_df[display_name] = values_pct

    display_df["1-year FEV1"] = raw_df["fev1_1y"].astype(float).map(lambda value: f"{value:.2f}")
    download_df["1-year FEV1"] = raw_df["fev1_1y"].astype(float)

    outcome_columns = [
        ("severe_ACR", "Severe ACR (%)"),
        ("ever_clinical_AMR", "Clinical AMR (%)"),
        ("BLAD", "BLAD (%)"),
    ]
    for raw_column, display_name in outcome_columns:
        values_pct = 100.0 * raw_df[raw_column].astype(float)
        display_df[display_name] = values_pct.map(lambda value: f"{value:.2f}")
        download_df[display_name] = values_pct

    return display_df, download_df


def _build_editor_controls(record: dict, schema, dropdown_options: dict[str, dict[str, object]]):
    category_columns = set(
        schema.passthrough_category_columns + schema.embedding_category_columns
    )
    controls = []

    for feature_name in schema.feature_names:
        label = FEATURE_LABEL_MAP.get(feature_name, feature_name)
        value = record.get(feature_name, "")

        if feature_name in category_columns:
            control = dcc.Dropdown(
                id={"type": "editor-input", "feature": feature_name},
                options=dropdown_options[feature_name]["options"],
                value=value if value not in {"", None} else None,
                clearable=True,
                placeholder="(empty)",
                className="editor-dropdown",
            )
        else:
            control = dbc.Input(
                id={"type": "editor-input", "feature": feature_name},
                type="number",
                step="any",
                value=value if value != "" else None,
                placeholder="(empty)",
                size="sm",
            )

        controls.append(
            dbc.Col(
                [
                    html.Div(label, className="metric-label", style={"marginBottom": "6px"}),
                    control,
                ],
                xs=12,
                md=6,
                xl=4,
                className="mb-3",
            )
        )

    return html.Div(
        [
            dbc.Alert("Blank value means missing.", color="light", className="py-2 mb-3"),
            dbc.Row(controls, className="g-2"),
        ]
    )


app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id="uploaded-data-store"),
        dcc.Store(id="selected-row-store"),
        dcc.Store(id="edited-row-store"),
        dcc.Download(id="download-prediction-table"),
        html.Div(
            className="hero",
            children=[
                html.Div("cfDNA Multitask DeepHit", className="hero-title"),
                html.P(
                    "Trial 506 app for 2-year mortality, 1-year FEV1, severe ACR, clinical AMR, BLAD, MC-dropout uncertainty, and SHAP.",
                    className="hero-sub",
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            className="soft-card mb-3",
                            children=[
                                dbc.CardBody(
                                    [
                                        html.Div("Input", className="section-title"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Load example data",
                                                        id="load-example-button",
                                                        color="success",
                                                        className="w-100",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Download example CSV",
                                                        id="download-example-button",
                                                        color="primary",
                                                        className="w-100",
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-2 mb-3",
                                        ),
                                        dcc.Download(id="download-example"),
                                        dcc.Upload(
                                            id="upload-data",
                                            className="upload-zone",
                                            multiple=False,
                                            children=html.Div(
                                                [
                                                    html.Div("Click or drag a cfDNA predictor CSV here"),
                                                    html.Small(
                                                        "Expected columns follow the trial 506 baseline feature schema."
                                                    ),
                                                ]
                                            ),
                                        ),
                                        html.Div(id="upload-status", className="mt-3"),
                                    ]
                                )
                            ],
                        ),
                        dbc.Card(
                            className="soft-card",
                            children=[
                                dbc.CardBody(
                                    [
                                        html.Div("Notes", className="section-title"),
                                        html.Ul(
                                            [
                                                html.Li("The app expects the trial 506 baseline predictors."),
                                                html.Li("No conformal calibration is used here; uncertainty comes from MC-dropout."),
                                                html.Li("Saved SHAP caches are reused when available; otherwise saved explainers are tried before live Kernel SHAP."),
                                            ],
                                            style={"paddingLeft": "18px", "marginBottom": 0},
                                        ),
                                    ]
                                )
                            ],
                        ),
                    ],
                    xs=12,
                    lg=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            className="soft-card mb-3",
                            children=[
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            [
                                                html.Span("Prediction table", className="section-title"),
                                                html.Div(
                                                    [
                                                        html.Span("Select one row", className="table-chip"),
                                                        dbc.Button(
                                                            "Download full table",
                                                            id="download-prediction-table-button",
                                                            color="outline-primary",
                                                            size="sm",
                                                            disabled=True,
                                                            className="ms-2",
                                                        ),
                                                    ],
                                                    style={"display": "flex", "alignItems": "center"},
                                                ),
                                            ],
                                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                        ),
                                        html.Div(id="prediction-table-wrap"),
                                    ]
                                )
                            ],
                        ),
                        dbc.Card(
                            className="soft-card",
                            children=[
                                dbc.CardBody(
                                    [
                                        html.Div("Selected patient", className="section-title"),
                                        html.Div(id="patient-banner", className="mb-3"),
                                        dbc.Row(id="metric-cards-row", className="g-3 mb-3"),
                                        dcc.Graph(id="survival-graph", config={"displayModeBar": False}),
                                        dcc.Graph(id="fev1-graph", config={"displayModeBar": False}),
                                        html.Hr(),
                                        html.Div("Editable feature row", className="section-title"),
                                        html.Div(id="editor-wrap"),
                                        dbc.Button(
                                            "Apply edits and rerun",
                                            id="apply-edit-button",
                                            color="warning",
                                            className="mt-3",
                                        ),
                                        html.Hr(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Dropdown(
                                                        id="shap-target-dropdown",
                                                        options=[
                                                            {"label": spec["label"], "value": key}
                                                            for key, spec in TARGET_SPECS.items()
                                                        ],
                                                        value="mortality_2y",
                                                        clearable=False,
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Checklist(
                                                        id="force-dynamic-shap",
                                                        options=[
                                                            {
                                                                "label": "Force live Kernel SHAP instead of saved cache/explainer",
                                                                "value": "force",
                                                            }
                                                        ],
                                                        value=[],
                                                        switch=True,
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="align-items-center",
                                        ),
                                        dbc.Button(
                                            "Render SHAP",
                                            id="render-shap-button",
                                            color="secondary",
                                            className="mt-3 mb-3",
                                        ),
                                        html.Div(id="shap-status", className="mb-2"),
                                        html.Div(id="shap-graph"),
                                    ]
                                )
                            ],
                        ),
                    ],
                    xs=12,
                    lg=9,
                ),
            ]
        ),
    ],
)


def parse_uploaded_csv(contents: str) -> pd.DataFrame:
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    if "subject_number" in df.columns and "SUBJECT_NUMBER" not in df.columns:
        df = df.rename(columns={"subject_number": "SUBJECT_NUMBER"})
    return df


def build_loaded_store(df: pd.DataFrame, source_name: str):
    service = get_prediction_service()
    subject_numbers = (
        df["SUBJECT_NUMBER"].astype(str).tolist()
        if "SUBJECT_NUMBER" in df.columns
        else None
    )
    batch = service.prepare_batch(df, subject_numbers=subject_numbers)
    prediction_df = service.predict_summary(batch)
    known_indices = [
        service.lookup_known_patient_index(subject) for subject in batch.subject_numbers
    ]
    store = {
        "raw_records": batch.raw_features.to_dict("records"),
        "subject_numbers": batch.subject_numbers,
        "predictions": prediction_df.to_dict("records"),
        "known_indices": known_indices,
        "warnings": batch.warnings,
    }
    pieces = [
        dbc.Alert(
            f"{source_name} loaded successfully with {len(batch.raw_features)} row(s).",
            color="success",
            className="mb-2",
        )
    ]
    if batch.warnings:
        pieces.append(
            dbc.Alert(
                "Preprocessing notes: " + " ".join(batch.warnings[:4]),
                color="warning",
                className="mb-0",
            )
        )
    return store, pieces


def load_example_dataframe() -> pd.DataFrame:
    schema = build_schema_artifacts()
    baseline = load_baseline_frame()
    return baseline[["SUBJECT_NUMBER", *schema.feature_names]].head(10).copy()


@app.callback(
    Output("download-example", "data"),
    Input("download-example-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_example(_n_clicks):
    schema = build_schema_artifacts()
    baseline = load_baseline_frame()
    example = baseline[["SUBJECT_NUMBER", *schema.feature_names]].head(10).copy()
    return dcc.send_data_frame(example.to_csv, "cfdna_trial506_example.csv", index=False)


@app.callback(
    Output("uploaded-data-store", "data"),
    Output("upload-status", "children"),
    Output("selected-row-store", "data"),
    Output("edited-row-store", "data"),
    Input("upload-data", "contents"),
    Input("load-example-button", "n_clicks"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, _load_example_clicks, filename):
    trigger = None
    if dash.callback_context.triggered:
        trigger = dash.callback_context.triggered[0]["prop_id"].split(".", 1)[0]

    try:
        if trigger == "load-example-button":
            store, pieces = build_loaded_store(load_example_dataframe(), "Bundled example data")
            return store, pieces, {"row_index": 0}, None
        if contents is None:
            return no_update, dbc.Alert("Upload a CSV to begin.", color="light"), no_update, no_update
        df = parse_uploaded_csv(contents)
        store, pieces = build_loaded_store(df, filename or "CSV")
        return store, pieces, {"row_index": 0}, None
    except Exception as exc:
        return no_update, dbc.Alert(str(exc), color="danger"), no_update, no_update


@app.callback(
    Output("download-prediction-table-button", "disabled"),
    Input("uploaded-data-store", "data"),
)
def toggle_prediction_download_button(store):
    return not bool(store and store.get("predictions"))


@app.callback(
    Output("download-prediction-table", "data"),
    Input("download-prediction-table-button", "n_clicks"),
    State("uploaded-data-store", "data"),
    prevent_initial_call=True,
)
def download_prediction_table(_n_clicks, store):
    if not store or not store.get("predictions"):
        raise dash.exceptions.PreventUpdate

    _display_df, download_df = build_prediction_frames(store["predictions"])
    return dcc.send_data_frame(
        download_df.to_csv,
        "cfdna_multitask_prediction_table.csv",
        index=False,
        float_format="%.6f",
    )


@app.callback(
    Output("prediction-table-wrap", "children"),
    Input("uploaded-data-store", "data"),
)
def render_prediction_table(store):
    if not store:
        return dbc.Alert("Predictions will appear here after upload.", color="light")

    df = pd.DataFrame(store["predictions"]).copy()
    if df.empty:
        return dbc.Alert("No rows were available for prediction.", color="warning")

    display_df, _download_df = build_prediction_frames(store["predictions"])

    return dash_table.DataTable(
        id="prediction-table",
        data=display_df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in display_df.columns],
        row_selectable="single",
        selected_rows=[0] if len(display_df) else [],
        style_table={"overflowX": "auto", "maxHeight": "460px"},
        style_cell={"padding": "10px", "fontSize": "0.88rem", "textAlign": "left", "minWidth": "120px"},
        style_header={
            "backgroundColor": "#102a43",
            "color": "white",
            "fontWeight": "600",
            "border": "0",
        },
        style_data={"backgroundColor": "rgba(255,255,255,0.78)", "border": "0"},
        style_as_list_view=True,
    )


@app.callback(
    Output("selected-row-store", "data"),
    Input("prediction-table", "selected_rows"),
    prevent_initial_call=True,
)
def store_selected_row(selected_rows):
    if not selected_rows:
        return no_update
    return {"row_index": int(selected_rows[0])}


@app.callback(
    Output("editor-wrap", "children"),
    Input("uploaded-data-store", "data"),
    Input("selected-row-store", "data"),
)
def render_editor(store, selected_row):
    if not store or not selected_row:
        return dbc.Alert("Select a patient row to edit features.", color="light")

    row_index = int(selected_row["row_index"])
    raw_records = store["raw_records"]
    if row_index >= len(raw_records):
        return dbc.Alert("Selected row is out of range.", color="danger")

    schema = build_schema_artifacts()
    dropdown_options = build_category_dropdown_options(schema)
    normalized_record = normalize_editor_record(raw_records[row_index], schema)
    return _build_editor_controls(normalized_record, schema, dropdown_options)


@app.callback(
    Output("edited-row-store", "data"),
    Input("apply-edit-button", "n_clicks"),
    State({"type": "editor-input", "feature": ALL}, "value"),
    State({"type": "editor-input", "feature": ALL}, "id"),
    State("selected-row-store", "data"),
    prevent_initial_call=True,
)
def apply_edits(_n_clicks, editor_values, editor_ids, selected_row):
    if not editor_ids or not selected_row:
        return no_update
    record = {
        item["feature"]: value
        for item, value in zip(editor_ids, editor_values)
    }
    schema = build_schema_artifacts()
    return {
        "row_index": int(selected_row["row_index"]),
        "record": coerce_editor_record_for_model(record, schema),
    }


def _build_detail_context(store: dict, selected_row: dict, edited_row: dict | None):
    service = get_prediction_service()
    row_index = int(selected_row["row_index"])
    subject_numbers = store["subject_numbers"]
    raw_records = store["raw_records"]
    subject_number = subject_numbers[row_index]

    original_batch = service.prepare_batch(
        pd.DataFrame([raw_records[row_index]]),
        subject_numbers=[subject_number],
    )
    original_detail = service.predict_single(original_batch, mc_samples=MC_SAMPLES_DEFAULT)

    updated_detail = None
    updated_batch = None
    if edited_row and int(edited_row["row_index"]) == row_index:
        updated_batch = service.prepare_batch(
            pd.DataFrame([edited_row["record"]]),
            subject_numbers=[subject_number],
        )
        updated_detail = service.predict_single(updated_batch, mc_samples=MC_SAMPLES_DEFAULT)

    known_index = store["known_indices"][row_index]
    return service, row_index, subject_number, known_index, original_detail, updated_detail, updated_batch


@app.callback(
    Output("patient-banner", "children"),
    Output("metric-cards-row", "children"),
    Output("survival-graph", "figure"),
    Output("fev1-graph", "figure"),
    Input("uploaded-data-store", "data"),
    Input("selected-row-store", "data"),
    Input("edited-row-store", "data"),
)
def render_patient_details(store, selected_row, edited_row):
    empty_fig = {}
    if not store or not selected_row:
        return (
            dbc.Alert("Select a patient in the prediction table to inspect details.", color="light"),
            [],
            empty_fig,
            empty_fig,
        )

    try:
        _service, row_index, subject_number, known_index, original_detail, updated_detail, _updated_batch = _build_detail_context(
            store, selected_row, edited_row
        )
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger"), [], empty_fig, empty_fig

    banner_children = [
        dbc.Alert(
            f"Selected subject: {subject_number}"
            + (f" | matched cohort index: {known_index}" if known_index is not None else ""),
            color="info",
            className="mb-2",
        )
    ]
    if original_detail["warnings"]:
        banner_children.append(
            dbc.Alert(" ".join(original_detail["warnings"]), color="warning", className="mb-2")
        )
    if original_detail.get("fev1_scaled_fallback"):
        banner_children.append(
            dbc.Alert(
                "FEV1 scaling metadata could not be loaded, so the app is showing standardized FEV1 units.",
                color="warning",
                className="mb-0",
            )
        )

    cards = [
        dbc.Col(metric_card(TARGET_SPECS[key]["label"], key, original_detail, updated_detail), md=6, xl=4)
        for key in TARGET_SPECS
    ]
    survival_fig = build_survival_figure(original_detail, updated_detail)
    fev1_fig = build_fev1_figure(original_detail, updated_detail)
    return banner_children, cards, survival_fig, fev1_fig


@app.callback(
    Output("shap-status", "children"),
    Output("shap-graph", "children"),
    Input("render-shap-button", "n_clicks"),
    State("shap-target-dropdown", "value"),
    State("force-dynamic-shap", "value"),
    State("uploaded-data-store", "data"),
    State("selected-row-store", "data"),
    State("edited-row-store", "data"),
    prevent_initial_call=True,
)
def render_shap(_n_clicks, target_key, force_flags, store, selected_row, edited_row):
    if not store or not selected_row:
        return dbc.Alert("Upload data and select a patient row first.", color="light"), {}

    try:
        service, _row_index, _subject_number, known_index, original_detail, updated_detail, _updated_batch = _build_detail_context(
            store, selected_row, edited_row
        )
        detail = updated_detail or original_detail
        patient_index = None if updated_detail is not None else known_index
        explanation = compute_individual_explanation(
            service,
            detail,
            target_key,
            patient_index=patient_index,
            force_recompute=("force" in (force_flags or [])),
        )
        source_text = {
            "saved_cache": "from saved cache.",
            "saved_explainer": "from saved explainer.",
            "dynamic_kernel_shap": "with live Kernel SHAP.",
        }.get(explanation.source, "with live Kernel SHAP.")
        status = dbc.Alert(
            f"SHAP rendered for {TARGET_SPECS[target_key]['label']} " + source_text,
            color="success" if explanation.source in {"saved_cache", "saved_explainer"} else "secondary",
        )
        png_bytes = build_waterfall_png_bytes(explanation, top_n=8)
        png_token = _store_shap_image(png_bytes)
        return status, html.Div(
            [
                html.Img(
                    src=build_waterfall_image_data_url(explanation, top_n=8),
                    style={
                        "width": "100%",
                        "height": "auto",
                        "display": "block",
                        "backgroundColor": "white",
                        "borderRadius": "10px",
                    },
                ),
                html.Div(
                    html.A(
                        "Open SHAP image directly",
                        href=f"/shap-image/{png_token}.png",
                        target="_blank",
                    ),
                    style={"marginTop": "8px"},
                ),
            ]
        )
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger"), html.Div()


if __name__ == "__main__":
    app.run(
        debug=(os.getenv("DASH_DEBUG", "0") == "1"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8052")),
    )
