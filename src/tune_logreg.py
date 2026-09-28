"""
tune_logreg.py
--------------
Now that Logistic Regression is the FINAL chosen model (it beat tuned
Random Forest on the test set -- see reports/evaluation_report.md),
it gets the same fair tuning treatment RF got in Step 6: GridSearchCV
scored on the validation fold only (PredefinedSplit), test set never
loaded into this script.

Tunes: C (inverse regularization strength) and class_weight.

Run:
    python src/tune_logreg.py
Outputs:
    models/final_model.joblib
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit


def load_train_val():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")["readiness_category"]
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv")["readiness_category"]
    return X_train, y_train, X_val, y_val


def main():
    X_train, y_train, X_val, y_val = load_train_val()
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
    ps = PredefinedSplit(test_fold)

    param_grid = {
        "C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
        "class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=2000, random_state=42),
        param_grid=param_grid,
        cv=ps,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_combined, y_combined)

    print(f"Best params: {grid.best_params_}")
    print(f"Best validation macro-F1: {grid.best_score_:.3f}")

    joblib.dump(grid.best_estimator_, "models/final_model.joblib")
    print("Saved models/final_model.joblib")


if __name__ == "__main__":
    main()
