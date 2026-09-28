"""
tune.py
-------
Tunes the Random Forest (the "stronger model") using ONLY the train
and validation splits. The test set is not imported into this script
at all -- that's a structural guarantee against test-set leakage
during tuning, not just a promise in a comment.

How the search respects the train/val roles (not k-fold blending):
    We use sklearn's PredefinedSplit so that for every candidate
    hyperparameter combination, GridSearchCV fits ONLY on the training
    rows and scores ONLY on the validation rows -- exactly mirroring
    the fixed split from Step 4, rather than doing generic k-fold CV
    on a train+val blend.

Why default RF underperformed (Step 5 finding) and what we're fixing:
    n_estimators=300 with unlimited depth overfit the 2,100-row
    training set and collapsed minority-class ("Highly Ready")
    predictions. This grid searches shallower trees, larger leaf
    sizes, and class_weight='balanced' to counter both the overfitting
    and the class imbalance.

Run:
    python src/tune.py
Outputs:
    models/random_forest_tuned.joblib  (refit on train+val with best params)
    printed best params + validation macro-F1
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
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

    # PredefinedSplit convention: -1 = always in the training fold,
    # 0 = the single validation fold used for scoring.
    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.full(len(X_val), 0),
    ])
    ps = PredefinedSplit(test_fold)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [4, 8, 12, None],
        "min_samples_leaf": [1, 5, 10],
        "class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=ps,
        scoring="f1_macro",  # macro-F1: treats all 4 classes equally, matches our imbalance concern
        n_jobs=-1,
        refit=True,  # after picking best params (scored on val), refit on train+val combined
    )

    print(f"Searching {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_leaf']) * len(param_grid['class_weight'])} combinations, scored on the validation fold only...")
    grid.fit(X_combined, y_combined)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best validation macro-F1: {grid.best_score_:.3f}")

    joblib.dump(grid.best_estimator_, "models/random_forest_tuned.joblib")
    print("\nSaved models/random_forest_tuned.joblib (refit on train+val with best params)")
    print("Test set was never loaded in this script -- see src/tune.py imports.")


if __name__ == "__main__":
    main()
