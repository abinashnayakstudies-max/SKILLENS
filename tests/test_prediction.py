"""
test_prediction.py
--------------------
Tests for src/predict.py: model loading, single/batch prediction, and
invalid-input handling (Step 12's robustness cases as real assertions).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from predict import load_artifacts, predict_from_raw, RAW_FEATURE_COLS, CATEGORY_ORDER


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


@pytest.fixture
def valid_row():
    df = pd.read_csv("data/raw/students.csv")
    return df[RAW_FEATURE_COLS].median().to_frame().T


def test_models_load_without_error(artifacts):
    pipeline, model = artifacts
    assert pipeline is not None
    assert model is not None


def test_predict_returns_valid_category(artifacts, valid_row):
    pipeline, model = artifacts
    result = predict_from_raw(valid_row, pipeline, model)
    assert result["predicted_category"].iloc[0] in CATEGORY_ORDER


def test_predict_score_in_valid_range(artifacts, valid_row):
    pipeline, model = artifacts
    result = predict_from_raw(valid_row, pipeline, model)
    score = result["readiness_score"].iloc[0]
    assert 0 <= score <= 100


def test_predict_missing_column_raises(artifacts, valid_row):
    pipeline, model = artifacts
    incomplete = valid_row.drop(columns=["cgpa"])
    with pytest.raises(ValueError):
        predict_from_raw(incomplete, pipeline, model)


def test_predict_handles_extreme_but_valid_inputs(artifacts, valid_row):
    """Extreme (but technically in-range) values shouldn't crash or
    produce an out-of-bounds score.
    """
    pipeline, model = artifacts
    extreme = valid_row.copy()
    extreme["cgpa"] = 10.0
    extreme["leetcode_solved"] = 800
    extreme["backlogs"] = 0
    result = predict_from_raw(extreme, pipeline, model)
    assert 0 <= result["readiness_score"].iloc[0] <= 100

    extreme_low = valid_row.copy()
    extreme_low["cgpa"] = 4.0
    extreme_low["leetcode_solved"] = 0
    extreme_low["backlogs"] = 10
    result_low = predict_from_raw(extreme_low, pipeline, model)
    assert 0 <= result_low["readiness_score"].iloc[0] <= 100


def test_predict_handles_missing_values_in_row(artifacts, valid_row):
    """A NaN in a numeric column should be imputed by the pipeline, not crash."""
    pipeline, model = artifacts
    row_with_na = valid_row.copy()
    row_with_na["resume_score"] = np.nan
    result = predict_from_raw(row_with_na, pipeline, model)
    assert result["predicted_category"].iloc[0] in CATEGORY_ORDER


def test_batch_prediction_multiple_rows(artifacts):
    pipeline, model = artifacts
    df = pd.read_csv("data/raw/students.csv")[RAW_FEATURE_COLS].head(10)
    result = predict_from_raw(df, pipeline, model)
    assert len(result) == 10
    assert result["predicted_category"].isin(CATEGORY_ORDER).all()
