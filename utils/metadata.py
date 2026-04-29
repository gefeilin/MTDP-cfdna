from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .config import (
    BASELINE_CSV_FILENAME,
    CATEGORY_COLUMNS,
    CATEGORY_VALUE_LABEL_MAP,
    DATA_DIR,
    EMBEDDING_MIN_CLASSES,
    EXCLUDED_PREDICTOR_COLUMNS,
    FEATURE_LABEL_MAP,
    FEATURE_LABEL_REVERSE_MAP,
    SCHEMA_METADATA_PATH,
)


POWER2_DISPLAY_FEATURES = {"AUC_ddcfDNA_Copies_1_30"}


def load_baseline_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / BASELINE_CSV_FILENAME).rename(
        columns={"log2_rdcfDNA_(Copies/mL)": "log2_rdcfDNA_.Copies.mL."}
    )
    df = df[df["event_time"] > 30].copy()
    df = df.sort_values("SUBJECT_NUMBER").reset_index(drop=True)
    return df


def canonicalize_category_value(value) -> str | None:
    if pd.isna(value):
        return None

    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric_value):
        numeric_value = float(numeric_value)
        if abs(numeric_value - round(numeric_value)) < 1e-8:
            return str(int(round(numeric_value)))
        return str(numeric_value)

    return str(value)


@dataclass(frozen=True)
class SchemaArtifacts:
    cohort_df: pd.DataFrame
    feature_names: list[str]
    numerical_columns: list[str]
    passthrough_category_columns: list[str]
    embedding_category_columns: list[str]
    num_numerical: int
    cat_cardinalities: list[int]
    numerical_mean: pd.Series
    numerical_std: pd.Series
    embedding_codebooks: dict[str, dict[object, int]]
    subject_to_index: dict[str, int]


def _category_sort_key(value: str) -> tuple[int, float | str]:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value):
        return (0, float(numeric_value))
    return (1, str(value))


def _category_label_lookup(feature_name: str) -> tuple[dict[str, str], dict[str, str]]:
    raw_feature_name = FEATURE_LABEL_REVERSE_MAP.get(feature_name, feature_name)
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}

    for raw_value, label in CATEGORY_VALUE_LABEL_MAP.get(raw_feature_name, {}).items():
        canonical_value = canonicalize_category_value(raw_value)
        if canonical_value is None:
            continue
        label_text = str(label)
        forward[canonical_value] = label_text
        reverse[label_text] = canonical_value

    return forward, reverse


def _category_missing_code(feature_name: str) -> str | None:
    raw_feature_name = FEATURE_LABEL_REVERSE_MAP.get(feature_name, feature_name)
    missing_code = canonicalize_category_value(-1)
    if missing_code is None:
        return None
    for raw_value in CATEGORY_VALUE_LABEL_MAP.get(raw_feature_name, {}):
        if canonicalize_category_value(raw_value) == missing_code:
            return missing_code
    return None


def build_schema_artifacts() -> SchemaArtifacts:
    df = load_baseline_frame()
    if not SCHEMA_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Schema metadata file is missing: {SCHEMA_METADATA_PATH}"
        )

    with SCHEMA_METADATA_PATH.open("r", encoding="utf-8") as handle:
        schema_payload = json.load(handle)

    passthrough_category_columns = list(schema_payload["passthrough_category_columns"])
    embedding_category_columns = list(schema_payload["embedding_category_columns"])
    numerical_columns = list(schema_payload["numerical_columns"])
    feature_names = list(schema_payload["feature_names"])
    numerical_mean = pd.Series(schema_payload["numerical_mean"], dtype="float64").reindex(
        numerical_columns
    )
    numerical_std = pd.Series(schema_payload["numerical_std"], dtype="float64").reindex(
        numerical_columns
    ).replace(0, 1.0)

    embedding_codebooks = {
        column: {str(key): int(value) for key, value in codebook.items()}
        for column, codebook in schema_payload["embedding_codebooks"].items()
    }
    cat_cardinalities = [int(value) for value in schema_payload["cat_cardinalities"]]

    subject_to_index = {
        str(subject): index for index, subject in enumerate(df["SUBJECT_NUMBER"].astype(str))
    }

    return SchemaArtifacts(
        cohort_df=df,
        feature_names=feature_names,
        numerical_columns=numerical_columns,
        passthrough_category_columns=passthrough_category_columns,
        embedding_category_columns=embedding_category_columns,
        num_numerical=int(schema_payload["num_numerical"]),
        cat_cardinalities=cat_cardinalities,
        numerical_mean=numerical_mean,
        numerical_std=numerical_std,
        embedding_codebooks=embedding_codebooks,
        subject_to_index=subject_to_index,
    )


def category_value_to_editor_label(feature_name: str, value) -> str:
    if pd.isna(value):
        return ""

    canonical_value = canonicalize_category_value(value)
    if canonical_value is None:
        return ""
    if canonical_value == _category_missing_code(feature_name):
        return ""

    forward, _reverse = _category_label_lookup(feature_name)
    return forward.get(canonical_value, canonical_value)


def editor_label_to_category_value(feature_name: str, value) -> str:
    if pd.isna(value) or value == "":
        missing_code = _category_missing_code(feature_name)
        return "" if missing_code is None else missing_code

    forward, reverse = _category_label_lookup(feature_name)
    value_text = str(value)
    if value_text in reverse:
        return reverse[value_text]

    canonical_value = canonicalize_category_value(value_text)
    if canonical_value is None:
        return ""
    if canonical_value in forward:
        return canonical_value
    return canonical_value


def build_category_dropdown_options(schema: SchemaArtifacts) -> dict[str, dict[str, object]]:
    dropdowns: dict[str, dict[str, object]] = {}
    category_columns = schema.passthrough_category_columns + schema.embedding_category_columns

    for column in category_columns:
        options: list[dict[str, str]] = [{"label": "(empty)", "value": ""}]
        seen_labels = {""}
        forward, _reverse = _category_label_lookup(column)
        missing_code = _category_missing_code(column)

        for canonical_value, label_text in forward.items():
            if canonical_value == missing_code:
                continue
            if label_text in seen_labels:
                continue
            options.append({"label": label_text, "value": label_text})
            seen_labels.add(label_text)

        observed_values: set[str] = set()
        if column in schema.embedding_codebooks:
            observed_values.update(str(key) for key in schema.embedding_codebooks[column].keys())
        if column in schema.cohort_df.columns:
            observed_values.update(
                canonical_value
                for canonical_value in schema.cohort_df[column].map(canonicalize_category_value).dropna().tolist()
            )

        for canonical_value in sorted(observed_values, key=_category_sort_key):
            if canonical_value == missing_code:
                continue
            label_text = forward.get(canonical_value, canonical_value)
            if label_text in seen_labels:
                continue
            options.append({"label": label_text, "value": label_text})
            seen_labels.add(label_text)

        dropdowns[column] = {"options": options, "clearable": True}

    return dropdowns


def normalize_editor_record(record: dict, schema: SchemaArtifacts) -> dict:
    normalized: dict = {}
    category_columns = set(schema.passthrough_category_columns + schema.embedding_category_columns)

    for feature_name in schema.feature_names:
        value = record.get(feature_name)
        if feature_name in category_columns:
            normalized[feature_name] = category_value_to_editor_label(feature_name, value)
        else:
            normalized[feature_name] = (
                "" if pd.isna(value) else feature_value_to_editor_value(feature_name, value)
            )

    return normalized


def coerce_editor_record_for_model(record: dict, schema: SchemaArtifacts) -> dict:
    coerced: dict = {}
    category_columns = set(schema.passthrough_category_columns + schema.embedding_category_columns)

    for feature_name in schema.feature_names:
        value = record.get(feature_name)
        if feature_name in category_columns:
            canonical_value = editor_label_to_category_value(feature_name, value)
            coerced[feature_name] = np.nan if canonical_value == "" else canonical_value
        else:
            coerced[feature_name] = (
                np.nan if value == "" else editor_value_to_model_value(feature_name, value)
            )

    return coerced


def _raw_feature_name(feature_name: str) -> str:
    return FEATURE_LABEL_REVERSE_MAP.get(feature_name, feature_name)


def _uses_power2_display(feature_name: str) -> bool:
    return _raw_feature_name(feature_name) in POWER2_DISPLAY_FEATURES


def _coerce_numeric(value) -> float | None:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return None
    numeric_value = float(numeric_value)
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def feature_value_to_editor_value(feature_name: str, value):
    numeric_value = _coerce_numeric(value)
    if numeric_value is None:
        return value
    if _uses_power2_display(feature_name):
        return round(float(np.power(2.0, numeric_value)))
    return value


def editor_value_to_model_value(feature_name: str, value):
    numeric_value = _coerce_numeric(value)
    if numeric_value is None:
        return value
    if _uses_power2_display(feature_name):
        return np.nan if numeric_value <= 0 else float(np.log2(numeric_value))
    return value


def map_categorical_value_for_display(feature_name: str, value) -> str | float | int:
    if pd.isna(value):
        return "NA"

    raw_feature_name = FEATURE_LABEL_REVERSE_MAP.get(feature_name, feature_name)
    value_map = CATEGORY_VALUE_LABEL_MAP.get(raw_feature_name)
    if value_map is None:
        return value

    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return value

    int_value = int(round(float(numeric_value)))
    if abs(float(numeric_value) - int_value) < 1e-8 and int_value in value_map:
        return value_map[int_value]
    return value


def format_feature_value_for_display(feature_name: str, value) -> str:
    if _uses_power2_display(feature_name):
        numeric_value = _coerce_numeric(value)
        if numeric_value is None:
            return "NA"
        return f"{np.power(2.0, numeric_value):.0f}"

    mapped_value = map_categorical_value_for_display(feature_name, value)
    if pd.isna(mapped_value):
        return "NA"
    if isinstance(mapped_value, (float, np.floating)):
        return f"{float(mapped_value):.2f}"
    return str(mapped_value)


def build_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=FEATURE_LABEL_MAP).copy()
    for column in renamed.columns:
        renamed[column] = renamed[column].map(
            lambda value, col=column: format_feature_value_for_display(col, value)
        )
    return renamed
