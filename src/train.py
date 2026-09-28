"""
train.py
--------
Trains TWO models on the training split only, with default
hyperparameters (tuning happens later, in Step 6, using train+val
only -- never the test set).

Baseline:  Logistic Regression (multinomial) -- fast, interpretable
           coefficients, a fair "can a simple linear model do this?"
           reference point.
Stronger:  Random Forest -- captures non-linear interactions (e.g.
           the CGPA/DSA synergy penalty baked into the synthetic
           target) and gives free feature_importances_.

Run:
    python src/train.py
Outputs:
    models/baseline_logreg.joblib
    models/random_forest.joblib
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

CATEGORY_ORDER = ["Needs Improvement", "Developing", "Placement Ready", "Highly Ready"]


def load_processed():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")["readiness_category"]
    return X_train, y_train


def main():
    X_train, y_train = load_processed()

    print(f"Training on {len(X_train)} rows, {X_train.shape[1]} processed features.")

    # --- Baseline: Logistic Regression ---
    # NOTE: sklearn >=1.5 removed the `multi_class` kwarg -- lbfgs now
    # handles multinomial softmax automatically for >2 classes.
    baseline = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )
    baseline.fit(X_train, y_train)
    joblib.dump(baseline, "models/baseline_logreg.joblib")
    print("Trained + saved: models/baseline_logreg.joblib")

    # --- Stronger: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/random_forest.joblib")
    print("Trained + saved: models/random_forest.joblib")

    print("\nBoth models trained with DEFAULT hyperparameters.")
    print("Tuning (GridSearchCV on train/val only) happens in Step 6 -- see src/tune.py.")


if __name__ == "__main__":
    main()
