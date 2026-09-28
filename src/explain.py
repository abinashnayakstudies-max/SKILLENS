"""
explain.py
----------
Step 7: explainability for the final model (tuned Logistic
Regression, models/final_model.joblib).

1. GLOBAL importance: permutation importance on the test set. Chosen
   over raw coefficients because it directly answers "how much does
   shuffling this feature hurt real predictive performance?" -- valid
   for any model type, and more honest than coefficient magnitude
   alone (which can be misleading if features have correlated scaled
   ranges).

2. INDIVIDUAL explanation: feature-ablation against the continuous
   readiness score (see explain_one_student_score below). Chosen over
   coefficient*value decomposition because it explains the SAME
   quantity (the 0-100 score) regardless of which of the 4 categories
   a student lands in -- a coefficient-based explanation of "why this
   student was predicted 'Needs Improvement'" would technically
   describe what pushed them toward THAT class, which is confusingly
   the opposite sign of what most people mean by "positive factor".

Run:
    PYTHONPATH=src python src/explain.py
    (needs src/ on the path since it cross-imports predict.py and
    data_preprocessing.py)
Outputs:
    reports/figures/08_global_importance.png
    printed per-student explanations for 3 example test students
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

CATEGORY_ORDER = ["Needs Improvement", "Developing", "Placement Ready", "Highly Ready"]


def load_test():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")["readiness_category"]
    return X_test, y_test


def clean_feature_name(name: str) -> str:
    # ColumnTransformer prefixes names like "num__cgpa" -- strip that
    # for human-readable output.
    return name.split("__")[-1]


def global_importance(model, X_test, y_test):
    print("Computing permutation importance on the test set (10 repeats)...")
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1_macro", n_jobs=-1
    )
    importance = pd.Series(result.importances_mean, index=[clean_feature_name(c) for c in X_test.columns])
    importance = importance.sort_values(ascending=False)

    print("\nTop 10 most important features (drop in macro-F1 when shuffled):")
    print(importance.head(10).round(4))

    plt.figure(figsize=(8, 6))
    importance.head(12).sort_values().plot(kind="barh", color="teal")
    plt.title("Global Feature Importance (Permutation, macro-F1 drop)")
    plt.xlabel("Mean decrease in macro-F1 when feature is shuffled")
    plt.tight_layout()
    plt.savefig("reports/figures/08_global_importance.png", dpi=120)
    plt.close()
    print("Saved reports/figures/08_global_importance.png")
    return importance


def explain_one_student_score(X_row_raw: pd.DataFrame, pipeline, model):
    """Feature-ablation explanation against the continuous readiness
    SCORE (not a single class's log-odds) -- this is what makes
    "positive"/"negative" mean the same thing regardless of which
    category the student landed in.

    Method: for each raw feature, replace it with the TRAINING-SET
    MEDIAN (a neutral population-average value), recompute the
    readiness score with everything else unchanged, and record the
    difference (actual_score - ablated_score). A feature whose real
    value pushed the score up relative to "an average student" gets a
    positive contribution; one that pulled it down gets negative.
    This is the same core idea SHAP uses (compare to a baseline), just
    computed by direct ablation, which is easy to defend in a viva:
    "we swap in the average value and see how much the score moves."
    """
    from predict import predict_from_raw, RAW_FEATURE_COLS

    baseline_medians = pd.read_csv("data/raw/students.csv")[RAW_FEATURE_COLS].median()

    actual_score = predict_from_raw(X_row_raw, pipeline, model)["readiness_score"].iloc[0]

    contributions = {}
    for col in RAW_FEATURE_COLS:
        ablated = X_row_raw.copy()
        ablated[col] = baseline_medians[col]
        ablated_score = predict_from_raw(ablated, pipeline, model)["readiness_score"].iloc[0]
        contributions[col] = actual_score - ablated_score

    ranked = sorted(contributions.items(), key=lambda t: -abs(t[1]))
    return ranked, actual_score


def main():
    model = joblib.load("models/final_model.joblib")
    X_test, y_test = load_test()
    feature_names = [clean_feature_name(c) for c in X_test.columns]

    global_importance(model, X_test, y_test)

    print(f"\n{'=' * 60}\nINDIVIDUAL EXPLANATIONS (feature-ablation vs. readiness score)\n{'=' * 60}")

    # Re-load RAW (unscaled) test rows so ablation swaps in raw medians correctly
    from data_preprocessing import load_raw, split_data
    pipeline = joblib.load("models/preprocessing_pipeline.joblib")
    df_raw = load_raw()
    _, _, X_test_raw, _, _, _ = split_data(df_raw)

    preds_cat = model.predict(X_test)
    example_positions = []
    for cat in ["Needs Improvement", "Placement Ready", "Highly Ready"]:
        matches = np.where(preds_cat == cat)[0]
        if len(matches) > 0:
            example_positions.append((matches[0], cat))

    for pos, predicted_class in example_positions:
        row_raw = X_test_raw.iloc[[pos]]
        ranked, score = explain_one_student_score(row_raw, pipeline, model)

        print(f"\n--- Student (test row {pos}) | Predicted: {predicted_class} | Readiness Score: {score:.1f}/100 ---")
        print("Top factors (vs. an average student):")
        for feat, contrib in ranked[:5]:
            direction = "increased" if contrib > 0 else "decreased"
            print(f"  {feat:<25} {direction} the readiness score by {abs(contrib):.2f} points")


if __name__ == "__main__":
    main()
