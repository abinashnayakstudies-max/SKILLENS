"""
test_app_logic.py
-------------------
Tests for src/csv_validation.py -- the CSV robustness logic used by
app.py's batch-upload tab (Step 9/12). Tested directly (no Streamlit
needed) since this logic was deliberately extracted into pure
functions for exactly this reason.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from csv_validation import validate_and_clean, CSVValidationError
from predict import RAW_FEATURE_COLS


@pytest.fixture
def good_row_dict():
    df = pd.read_csv("data/raw/students.csv")
    return df[RAW_FEATURE_COLS].median().to_dict()


def test_empty_csv_raises(good_row_dict):
    empty_df = pd.DataFrame(columns=RAW_FEATURE_COLS)
    with pytest.raises(CSVValidationError):
        validate_and_clean(empty_df)


def test_missing_required_column_raises(good_row_dict):
    df = pd.DataFrame([good_row_dict]).drop(columns=["cgpa"])
    with pytest.raises(CSVValidationError):
        validate_and_clean(df)


def test_extra_column_is_reported_not_fatal(good_row_dict):
    row = dict(good_row_dict)
    row["student_name"] = "Alex"  # extra column, should NOT raise
    df = pd.DataFrame([row])
    cleaned, extra_cols, coercion_report, total_missing = validate_and_clean(df)
    assert "student_name" in extra_cols
    assert len(cleaned) == 1


def test_bad_text_value_is_coerced_not_fatal(good_row_dict):
    row = dict(good_row_dict)
    row["cgpa"] = "not-a-number"
    df = pd.DataFrame([row])
    cleaned, extra_cols, coercion_report, total_missing = validate_and_clean(df)
    assert coercion_report.get("cgpa") == 1
    assert pd.isna(cleaned["cgpa"].iloc[0])


def test_text_yes_no_internship_is_normalized(good_row_dict):
    row = dict(good_row_dict)
    row["has_internship"] = "Yes"
    df = pd.DataFrame([row])
    cleaned, _, _, _ = validate_and_clean(df)
    assert cleaned["has_internship"].iloc[0] == 1

    row2 = dict(good_row_dict)
    row2["has_internship"] = "no"
    df2 = pd.DataFrame([row2])
    cleaned2, _, _, _ = validate_and_clean(df2)
    assert cleaned2["has_internship"].iloc[0] == 0


def test_missing_values_are_counted(good_row_dict):
    row = dict(good_row_dict)
    row["resume_score"] = None
    row["communication_score"] = None
    df = pd.DataFrame([row])
    cleaned, extra_cols, coercion_report, total_missing = validate_and_clean(df)
    assert total_missing == 2


def test_multiple_rows_with_mixed_issues(good_row_dict):
    """A realistic messy CSV: one clean row, one with a bad value, one
    with a missing value -- none of it should raise.
    """
    r1 = dict(good_row_dict)
    r2 = dict(good_row_dict)
    r2["cgpa"] = "oops"
    r3 = dict(good_row_dict)
    r3["backlogs"] = None
    df = pd.DataFrame([r1, r2, r3])
    cleaned, extra_cols, coercion_report, total_missing = validate_and_clean(df)
    assert len(cleaned) == 3
    assert total_missing >= 1
