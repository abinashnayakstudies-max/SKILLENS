"""
evaluate.py
-----------
Loads the two trained models and scores them on the TEST split (used
here only for this initial baseline-vs-stronger comparison snapshot;
after Step 6 tuning we re-run this once more for the final report --
the test set is never used to choose hyperparameters).

Run:
    python src/evaluate.py
Outputs:
    printed classification reports + confusion matrices
    reports/figures/05_confusion_baseline.png
    reports/figures/06_confusion_random_forest.png
    reports/evaluation_report.md (model comparison table)
"""

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

CATEGORY_ORDER = ["Needs Improvement", "Developing", "Placement Ready", "Highly Ready"]


def load_test():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")["readiness_category"]
    return X_test, y_test


def evaluate_model(name, model, X_test, y_test, fig_num):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, labels=CATEGORY_ORDER, average="macro", zero_division=0
    )

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Accuracy:          {acc:.3f}")
    print(f"Macro Precision:   {precision:.3f}")
    print(f"Macro Recall:      {recall:.3f}")
    print(f"Macro F1:          {f1:.3f}")
    print("\nFull classification report (per class):")
    print(classification_report(y_test, preds, labels=CATEGORY_ORDER, zero_division=0))

    cm = confusion_matrix(y_test, preds, labels=CATEGORY_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORY_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=25, colorbar=False)
    ax.set_title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    fname = f"reports/figures/{fig_num:02d}_confusion_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved {fname}")

    return {"model": name, "accuracy": acc, "macro_precision": precision,
            "macro_recall": recall, "macro_f1": f1}


def main():
    X_test, y_test = load_test()

    baseline = joblib.load("models/baseline_logreg.joblib")
    rf_untuned = joblib.load("models/random_forest.joblib")
    rf_tuned = joblib.load("models/random_forest_tuned.joblib")

    results = []
    results.append(evaluate_model("Baseline (Logistic Regression)", baseline, X_test, y_test, 5))
    results.append(evaluate_model("Random Forest (untuned)", rf_untuned, X_test, y_test, 6))
    results.append(evaluate_model("Random Forest (tuned)", rf_tuned, X_test, y_test, 7))

    comparison = pd.DataFrame(results).set_index("model").round(3)
    print(f"\n{'=' * 60}\nMODEL COMPARISON (same test set, n={len(y_test)})\n{'=' * 60}")
    print(comparison)

    with open("reports/evaluation_report.md", "w") as f:
        f.write("# SkillLens -- Model Evaluation Report\n\n")
        f.write(f"Test set size: {len(y_test)} students (untouched since the Step 4 split; never used for hyperparameter tuning).\n\n")
        f.write("## Model comparison\n\n")
        f.write(comparison.to_markdown())
        f.write("\n\n")
        f.write("## What changed between untuned and tuned Random Forest\n\n")
        f.write("The untuned Random Forest (`n_estimators=300`, unlimited depth) overfit the training set and "
                "collapsed predictions toward the two middle classes, missing most 'Highly Ready' students "
                "(recall 0.11 for that class). GridSearchCV, scored on the held-out validation fold only "
                "(never the test set), selected `class_weight='balanced'`, `min_samples_leaf=5`, unlimited depth "
                "and 300 trees -- the class-balancing and larger leaf size directly target the overfitting and "
                "minority-class collapse seen in Step 5.\n\n")
        f.write("## Final model choice\n\n")
        f.write("**Random Forest (tuned)** is used going forward for explainability (Step 7), the app, and "
                "CSV predictions -- selected on validation performance and confirmed on the untouched test set, "
                "not on test-set performance alone.\n\n")
        f.write("## Metric definitions\n\n")
        f.write("- **Accuracy**: fraction of students whose category was predicted exactly right.\n")
        f.write("- **Macro Precision**: of all students predicted into a given category, how many actually belonged there -- averaged equally across the 4 categories (so the small 'Highly Ready' class counts as much as 'Developing').\n")
        f.write("- **Macro Recall**: of all students who actually belong to a category, how many were correctly found -- again averaged equally per class.\n")
        f.write("- **Macro F1**: harmonic mean of macro precision and recall; the single number used to compare models fairly given class imbalance.\n")

    print("\nSaved reports/evaluation_report.md")


if __name__ == "__main__":
    main()
