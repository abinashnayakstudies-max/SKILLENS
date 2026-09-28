"""
test_preprocessing.py
-----------------------
Tests for the preprocessing pipeline (src/data_preprocessing.py).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import build_pipeline, NUMERIC_COLS


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 50
    data = {col: rng.normal(50, 10, n) for col in NUMERIC_COLS}
    df = pd.DataFrame(data)
    return df


def test_pipeline_fits_without_error(sample_df):
    pipeline = build_pipeline()
    pipeline.fit(sample_df)  # should not raise


def test_pipeline_output_shape(sample_df):
    pipeline = build_pipeline()
    pipeline.fit(sample_df)
    transformed = pipeline.transform(sample_df)
    assert transformed.shape[0] == len(sample_df)
    assert transformed.shape[1] == len(NUMERIC_COLS)  # one-to-one, no one-hot expansion in v1


def test_pipeline_handles_missing_values(sample_df):
    df_with_na = sample_df.copy()
    df_with_na.loc[0:4, "cgpa"] = np.nan
    pipeline = build_pipeline()
    pipeline.fit(df_with_na)
    transformed = pipeline.transform(df_with_na)
    assert not np.isnan(transformed).any(), "Imputer should remove all NaNs"


def test_pipeline_scales_to_roughly_zero_mean(sample_df):
    pipeline = build_pipeline()
    pipeline.fit(sample_df)
    transformed = pipeline.transform(sample_df)
    means = transformed.mean(axis=0)
    assert np.allclose(means, 0, atol=1e-6), "StandardScaler should center training data at ~0"


def test_pipeline_is_reusable_on_new_data(sample_df):
    """Fit on one set, transform a DIFFERENT set (simulates train-fit,
    test-transform) -- the whole point of the pipeline design.
    """
    pipeline = build_pipeline()
    pipeline.fit(sample_df)

    rng = np.random.default_rng(1)
    new_df = pd.DataFrame({col: rng.normal(50, 10, 10) for col in NUMERIC_COLS})
    transformed = pipeline.transform(new_df)
    assert transformed.shape == (10, len(NUMERIC_COLS))
