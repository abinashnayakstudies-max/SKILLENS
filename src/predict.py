"""
predict.py
----------
THE single canonical scoring function for SkillLens. Every other part
of the system -- the Streamlit single-student page, CSV batch upload,
and the what-if simulator (Step 8) -- calls `predict_from_raw()` in
this file. Having one function means the score a user sees in the
what-if tool is guaranteed to be computed the same way as the score
in a CSV export.

Why a continuous 0-100 score exists alongside the 4-way category:
    The classifier's raw output is a probability for each of the 4
    ordered categories. We convert that into a smooth 0-100 score by
    taking a probability-weighted average of each category's score
    band midpoint (Needs Improvement=0-40 -> 20, Developing=40-60 ->
    50, Placement Ready=60-80 -> 70, Highly Ready=80-100 -> 90). This
    is a real, deterministic function of the model's own output --
    not a separate invented number.

    The category shown to the user is the classifier's own argmax
    decision (not re-derived from the score), so it always matches
    what the model actually predicted. The continuous score and the
    category can occasionally sit in slightly different bands near a
    boundary (e.g. score=61 with category="Developing") -- this is
    normal, expected behavior for a probabilistic classifier and is
    documented here rather than silently forced to agree.
"""

import pandas as pd
import joblib

CATEGORY_ORDER = ["Needs Improvement", "Developing", "Placement Ready", "Highly Ready"]
CATEGORY_MIDPOINTS = {
    "Needs Improvement": 20,
    "Developing": 50,
    "Placement Ready": 70,
    "Highly Ready": 90,
}

PIPELINE_PATH = "models/preprocessing_pipeline.joblib"
MODEL_PATH = "models/final_model.joblib"

RAW_FEATURE_COLS = [
    "cgpa", "tenth_pct", "twelfth_pct", "backlogs",
    "leetcode_solved", "contest_rating", "dsa_proficiency",
    "num_languages_known", "dbms_score", "oop_score", "os_score", "cn_score",
    "num_projects", "num_ml_projects", "github_commits",
    "has_internship", "internship_months",
    "aptitude_score", "communication_score", "resume_score", "interview_prep_score",
    "certifications_count", "extracurricular_score",
]


def load_artifacts():
    pipeline = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    return pipeline, model


def score_from_probs(probs: dict) -> float:
    """probs: {category_name: probability}. Returns weighted 0-100 score."""
    return sum(probs[cat] * CATEGORY_MIDPOINTS[cat] for cat in CATEGORY_ORDER)


def predict_from_raw(raw_df: pd.DataFrame, pipeline, model) -> pd.DataFrame:
    """raw_df: DataFrame with RAW_FEATURE_COLS (unscaled, as a user would
    enter them). Returns a DataFrame with predicted_category,
    readiness_score, and per-class probabilities appended.

    This function is intentionally the ONLY place predict_proba is
    called on the final model -- reuse it, don't reimplement it.
    """
    missing = set(RAW_FEATURE_COLS) - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = raw_df[RAW_FEATURE_COLS]
    X_transformed = pipeline.transform(X)
    X_transformed = pd.DataFrame(X_transformed, columns=pipeline.get_feature_names_out(), index=raw_df.index)
    proba = model.predict_proba(X_transformed)
    classes = list(model.classes_)

    results = raw_df.copy()
    for i, cls in enumerate(classes):
        results[f"prob_{cls.replace(' ', '_')}"] = proba[:, i]

    results["predicted_category"] = [classes[i] for i in proba.argmax(axis=1)]
    results["readiness_score"] = [
        score_from_probs({cls: proba[row_i, classes.index(cls)] for cls in CATEGORY_ORDER})
        for row_i in range(len(raw_df))
    ]
    return results


if __name__ == "__main__":
    # smoke test on one row of raw test-like data
    pipeline, model = load_artifacts()
    sample = pd.DataFrame([{
        "cgpa": 8.2, "tenth_pct": 90, "twelfth_pct": 85, "backlogs": 0,
        "leetcode_solved": 220, "contest_rating": 1500, "dsa_proficiency": 7,
        "num_languages_known": 4, "dbms_score": 4, "oop_score": 4, "os_score": 3, "cn_score": 3,
        "num_projects": 4, "num_ml_projects": 2, "github_commits": 300,
        "has_internship": 1, "internship_months": 3,
        "aptitude_score": 72, "communication_score": 68, "resume_score": 65, "interview_prep_score": 60,
        "certifications_count": 2, "extracurricular_score": 55,
    }])
    out = predict_from_raw(sample, pipeline, model)
    print(out[["predicted_category", "readiness_score"]].to_string())
    print(out.filter(like="prob_").to_string())
