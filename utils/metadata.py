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
