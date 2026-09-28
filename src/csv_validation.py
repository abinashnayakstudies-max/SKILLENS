"""
csv_validation.py
-------------------
Pure, testable functions for validating and cleaning an uploaded CSV
before prediction. Used by both app.py (Step 9's CSV tab) and the
pytest suite (Step 13) -- keeping this logic out of app.py means it
can be unit-tested without running Streamlit.
"""

import pandas as pd

from predict import RAW_FEATURE_COLS


class CSVValidationError(Exception):
    """Raised for problems that must stop processing (empty file,
    missing required columns) -- as opposed to problems we can repair
    automatically (bad values, unseen internship text, extra columns).
    """
    pass


def check_not_empty(df: pd.DataFrame):
    if df.shape[0] == 0:
        raise CSVValidationError("The uploaded CSV has no data rows.")


def check_required_columns(df: pd.DataFrame, required=RAW_FEATURE_COLS):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise CSVValidationError(f"Missing required column(s): {', '.join(missing)}")


def find_extra_columns(df: pd.DataFrame, required=RAW_FEATURE_COLS):
    return [c for c in df.columns if c not in required]


def normalize_has_internship(series: pd.Series) -> pd.Series:
    """Accepts 0/1, or common Yes/No text variants. Anything else
    becomes NaN (caught by the numeric-missing-value path downstream,
    not a crash).

    Uses pd.api.types.is_numeric_dtype rather than `dtype == object`:
    newer pandas can infer a column of strings as its dedicated `str`
    dtype rather than legacy `object`, which `== object` would miss.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    return (
        series.astype(str).str.strip().str.lower()
        .map({"yes": 1, "y": 1, "true": 1, "1": 1,
              "no": 0, "n": 0, "false": 0, "0": 0})
    )


def coerce_numeric_columns(df: pd.DataFrame, cols=RAW_FEATURE_COLS):
    """Coerces each column to numeric, turning unparseable values into
    NaN instead of raising. Returns (df, report) where report maps
    column -> count of values that became newly invalid.
    """
    df = df.copy()
    report = {}
    for col in cols:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_na = df[col].isna().sum()
        newly_invalid = int(after_na - before_na)
        if newly_invalid > 0:
            report[col] = newly_invalid
    return df, report


def validate_and_clean(df: pd.DataFrame):
    """Full pipeline: raises CSVValidationError for unrecoverable
    problems (empty file, missing columns); otherwise returns
    (cleaned_df, extra_cols, coercion_report, total_missing) ready for
    predict_from_raw().
    """
    check_not_empty(df)
    check_required_columns(df)

    extra_cols = find_extra_columns(df)

    working = df.copy()
    if "has_internship" in working.columns:
        working["has_internship"] = normalize_has_internship(working["has_internship"])

    working, coercion_report = coerce_numeric_columns(working, RAW_FEATURE_COLS)
    total_missing = int(working[RAW_FEATURE_COLS].isna().sum().sum())

    return working, extra_cols, coercion_report, total_missing
