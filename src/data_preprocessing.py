"""
data_preprocessing.py
----------------------
Builds ONE reusable sklearn preprocessing pipeline and a fixed,
stratified 70/15/15 train/validation/test split.

Why one pipeline object:
    The exact same fitted ColumnTransformer is used for training,
    validation, testing, the Streamlit single-prediction page, the
    CSV batch upload, and the what-if simulator. This guarantees
    train/serve consistency and is the #1 thing to point to in a viva
    when asked "how do you avoid train-serve skew?".

Run:
    python src/data_preprocessing.py
Outputs:
    data/processed/X_train.csv, X_val.csv, X_test.csv
    data/processed/y_train.csv, y_val.csv, y_test.csv
    models/preprocessing_pipeline.joblib
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CATEGORY_ORDER = ["Needs Improvement", "Developing", "Placement Ready", "Highly Ready"]

TARGET_COL = "readiness_category"

# Columns by type. `has_internship` is already 0/1 so it's treated as
# numeric (no encoding needed); if you add true string categoricals
# later (e.g. "branch"), list them in CATEGORICAL_COLS.
NUMERIC_COLS = [
    "cgpa", "tenth_pct", "twelfth_pct", "backlogs",
    "leetcode_solved", "contest_rating", "dsa_proficiency",
    "num_languages_known", "dbms_score", "oop_score", "os_score", "cn_score",
    "num_projects", "num_ml_projects", "github_commits",
    "has_internship", "internship_months",
    "aptitude_score", "communication_score", "resume_score", "interview_prep_score",
    "certifications_count", "extracurricular_score",
]
CATEGORICAL_COLS = []  # placeholder: none in v1, kept for extensibility

ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def build_pipeline() -> ColumnTransformer:
    """Numeric: median-impute (robust to outliers) then standard-scale.
    Categorical: most-frequent-impute then one-hot (handle_unknown=
    'ignore' so an unseen category at prediction time doesn't crash the
    app -- it just becomes an all-zero row instead of erroring).
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [("num", numeric_pipeline, NUMERIC_COLS)]
    if CATEGORICAL_COLS:
        transformers.append(("cat", categorical_pipeline, CATEGORICAL_COLS))

    return ColumnTransformer(transformers=transformers)


def load_raw(path="data/raw/students.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET_COL] = pd.Categorical(df[TARGET_COL], categories=CATEGORY_ORDER, ordered=True)
    return df


def split_data(df: pd.DataFrame, seed: int = 42):
    """Fixed 70/15/15 split, stratified on the target so each split has
    the same class proportions as the full dataset. Two-step split
    (train vs rest, then rest -> val/test) because sklearn only splits
    two ways at a time.
    """
    X = df[ALL_FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def check_no_leakage(X_train, X_val, X_test):
    """Sanity check: no row (by index) appears in more than one split."""
    idx_train, idx_val, idx_test = set(X_train.index), set(X_val.index), set(X_test.index)
    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)
    print("Leakage check passed: train/val/test indices are disjoint.")


def main():
    df = load_raw()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    check_no_leakage(X_train, X_val, X_test)

    print(f"Train: {len(X_train)} ({len(X_train)/len(df):.1%})")
    print(f"Val:   {len(X_val)} ({len(X_val)/len(df):.1%})")
    print(f"Test:  {len(X_test)} ({len(X_test)/len(df):.1%})")

    print("\nClass balance check (should be similar across splits):")
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"  {name}: {y.value_counts(normalize=True).round(3).sort_index().to_dict()}")

    # Fit the pipeline on TRAINING DATA ONLY -- this is the critical
    # rule that prevents leakage: val/test statistics must never
    # influence the imputer medians or scaler mean/std.
    pipeline = build_pipeline()
    pipeline.fit(X_train)

    def transform_to_df(X, fitted_pipeline):
        arr = fitted_pipeline.transform(X)
        cols = fitted_pipeline.get_feature_names_out()
        return pd.DataFrame(arr, columns=cols, index=X.index)

    X_train_t = transform_to_df(X_train, pipeline)
    X_val_t = transform_to_df(X_val, pipeline)
    X_test_t = transform_to_df(X_test, pipeline)

    X_train_t.to_csv("data/processed/X_train.csv", index=False)
    X_val_t.to_csv("data/processed/X_val.csv", index=False)
    X_test_t.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    joblib.dump(pipeline, "models/preprocessing_pipeline.joblib")
    print("\nSaved processed splits to data/processed/ and pipeline to models/preprocessing_pipeline.joblib")


if __name__ == "__main__":
    main()
