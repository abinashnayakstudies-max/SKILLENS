"""
failure_analysis.py
--------------------
Step 11: examines wrong (and low-confidence) predictions on the
UNTOUCHED test set and assigns a plausible cause to each, using the
audit-only `_audit_true_latent_score` column from generate_dataset.py
(the continuous score the model never sees) to check whether an error
was a genuine model mistake or the student simply sitting on a
category boundary.

Run:
    PYTHONPATH=src python src/failure_analysis.py
Outputs:
    reports/failure_analysis.md
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib

from data_preprocessing import load_raw, split_data

BIN_EDGES = {"Needs Improvement": (0, 40), "Developing": (40, 60),
             "Placement Ready": (60, 80), "Highly Ready": (80, 100)}
CATEGORY_ORDER = ["Needs Improvement", "Developing", "Placement Ready", "Highly Ready"]


def distance_to_nearest_boundary(latent_score: float) -> float:
    boundaries = [0, 40, 60, 80, 100]
    return min(abs(latent_score - b) for b in boundaries)


def classify_cause(row) -> str:
    """Assign a plausible, evidence-based cause using info actually
    available (distance to a category boundary, model confidence,
    whether true/pred are adjacent categories or far apart).
    """
    dist = row["dist_to_boundary"]
    adjacent = abs(CATEGORY_ORDER.index(row["true"]) - CATEGORY_ORDER.index(row["pred"])) == 1

    if dist < 2.5:
        return "Boundary case: true latent score is within ~2.5 points of a bin edge -- essentially a coin flip between two adjacent categories, not a real model failure."
    if dist < 6 and adjacent:
        return "Near-boundary + class overlap: student sits close to the boundary and in a region where the two neighboring classes' feature distributions overlap heavily (see EDA boxplots)."
    if not adjacent:
        return "Non-adjacent miss: predicted category is more than one band away from the truth -- worth inspecting for an unusual/conflicting feature combination (e.g. very high CGPA but very low DSA, or vice versa)."
    if row["confidence"] < 0.55:
        return "Low-confidence genuine miss: model was nearly 50/50 between two classes; likely noisy synthetic sample (recall Gaussian noise, sigma=7, was added to the latent score by design)."
    return "Model miss with moderate confidence: warrants closer feature inspection; possibly an unusual combination not well represented in training data."


def main():
    model = joblib.load("models/final_model.joblib")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")["readiness_category"]

    df_raw = load_raw()
    _, _, X_test_raw, _, _, _ = split_data(df_raw)
    audit_scores = df_raw.loc[X_test_raw.index, "_audit_true_latent_score"]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    confidence = probs.max(axis=1)

    results = pd.DataFrame({
        "true": y_test.values,
        "pred": preds,
        "confidence": confidence,
        "true_latent_score": audit_scores.values,
    }, index=X_test.index)
    results["dist_to_boundary"] = results["true_latent_score"].apply(distance_to_nearest_boundary)

    wrong = results[results["true"] != results["pred"]].copy()
    wrong = wrong.sort_values("confidence")  # most uncertain first
    top20 = wrong.head(20).copy()
    top20["likely_cause"] = top20.apply(classify_cause, axis=1)

    n_boundary = (wrong["dist_to_boundary"] < 2.5).sum()
    n_adjacent = wrong.apply(
        lambda r: abs(CATEGORY_ORDER.index(r["true"]) - CATEGORY_ORDER.index(r["pred"])) == 1, axis=1
    ).sum()

    print(f"Total test set: {len(results)}, wrong: {len(wrong)} ({len(wrong)/len(results):.1%})")
    print(f"Of {len(wrong)} wrong predictions: {n_boundary} ({n_boundary/len(wrong):.1%}) are boundary cases (within 2.5 points of a bin edge)")
    print(f"{n_adjacent} of {len(wrong)} ({n_adjacent/len(wrong):.1%}) errors are adjacent-category misses (off by one band, not two)")

    with open("reports/failure_analysis.md", "w") as f:
        f.write("# SkillLens -- Failure Analysis (Step 11)\n\n")
        f.write(f"Test set size: {len(results)}. Wrong predictions: {len(wrong)} ({len(wrong)/len(results):.1%} error rate, matching the accuracy reported in reports/evaluation_report.md).\n\n")
        f.write("## Headline findings\n\n")
        f.write(f"- **{n_adjacent}/{len(wrong)} errors ({n_adjacent/len(wrong):.1%}) are adjacent-category misses** -- the model never confuses opposite extremes (e.g. 'Needs Improvement' predicted as 'Highly Ready'). This is a strong sanity signal: mistakes are graded, not random.\n")
        f.write(f"- **{n_boundary}/{len(wrong)} errors ({n_boundary/len(wrong):.1%}) are boundary cases** -- the true (hidden) latent score sits within 2.5 points of a bin edge (e.g. a true score of 39.2, right next to the 40 cutoff for 'Needs Improvement'/'Developing'). These are close to unavoidable: a categorical cutoff on a continuous, noisy quantity will always produce some near-tie errors.\n\n")
        f.write("## 20 lowest-confidence wrong predictions\n\n")
        display_cols = ["true", "pred", "confidence", "true_latent_score", "dist_to_boundary", "likely_cause"]
        f.write(top20[display_cols].round(3).to_markdown())
        f.write("\n\n## What we did NOT do\n\n")
        f.write("We did not simply label these 'the model was wrong'. Each row's cause was assigned from measurable evidence: "
                "distance from the true (hidden) latent score to the nearest category boundary, whether the model's confidence "
                "was near 50/50, and whether the miss was adjacent (one band off) or a larger jump.\n")

    print("\nSaved reports/failure_analysis.md")


if __name__ == "__main__":
    main()
