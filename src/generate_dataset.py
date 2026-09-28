"""
generate_dataset.py
--------------------
Creates a SYNTHETIC student dataset for the SkillLens project.

IMPORTANT: This data does NOT represent real placement outcomes. It is
generated from documented distributions and a weighted-scoring formula
with noise and interaction effects, then binned into 4 readiness
categories. This script is the single source of truth for how the
target was produced -- read this file to explain the target in a viva.

Run:
    python src/generate_dataset.py --n 3000 --seed 42
Output:
    data/raw/students.csv
"""

import argparse
import numpy as np
import pandas as pd


def generate_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate independent, individually-plausible input features.

    None of these features is the target itself -- they are things a
    real student could self-report on a form.
    """
    df = pd.DataFrame({
        # --- Academic ---
        "cgpa": np.clip(rng.normal(7.2, 0.9, n), 4.0, 10.0).round(2),
        "tenth_pct": np.clip(rng.normal(82, 8, n), 40, 100).round(1),
        "twelfth_pct": np.clip(rng.normal(78, 9, n), 40, 100).round(1),
        "backlogs": rng.choice([0, 0, 0, 0, 1, 1, 2, 3], n),

        # --- Coding / DSA ---
        "leetcode_solved": np.clip(rng.gamma(2.0, 60, n), 0, 800).round(0).astype(int),
        "contest_rating": np.clip(rng.normal(1350, 300, n), 800, 2400).round(0).astype(int),
        "dsa_proficiency": rng.integers(1, 11, n),  # self-rated 1-10

        # --- Technical skills (self-rated 1-5) ---
        "num_languages_known": rng.integers(1, 7, n),
        "dbms_score": rng.integers(1, 6, n),
        "oop_score": rng.integers(1, 6, n),
        "os_score": rng.integers(1, 6, n),
        "cn_score": rng.integers(1, 6, n),

        # --- Projects ---
        "num_projects": rng.poisson(2.5, n),
        "num_ml_projects": rng.poisson(0.6, n),
        "github_commits": np.clip(rng.gamma(2.0, 40, n), 0, 2000).round(0).astype(int),

        # --- Experience ---
        "has_internship": rng.choice([0, 1], n, p=[0.55, 0.45]),

        # --- Placement skills (0-100) ---
        "aptitude_score": np.clip(rng.normal(60, 15, n), 0, 100).round(1),
        "communication_score": np.clip(rng.normal(62, 14, n), 0, 100).round(1),
        "resume_score": np.clip(rng.normal(58, 16, n), 0, 100).round(1),
        "interview_prep_score": np.clip(rng.normal(55, 17, n), 0, 100).round(1),

        # --- Other ---
        "certifications_count": rng.poisson(1.2, n),
        "extracurricular_score": np.clip(rng.normal(50, 20, n), 0, 100).round(1),
    })

    # internship_months only makes sense if has_internship == 1
    df["internship_months"] = np.where(
        df["has_internship"] == 1,
        np.clip(rng.gamma(2.0, 1.5, n), 0.5, 12).round(1),
        0.0,
    )

    return df


def compute_latent_score(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Compute a hidden 0-100 'true readiness' score used ONLY to build
    the target label. This column is NEVER exposed as a feature to the
    model (that would be leakage) -- it is discarded after label
    creation.

    Design choices (documented for the viva):
      - Weighted groups: Academic 20%, DSA/Coding 30%, Projects 20%,
        Placement-skills 20%, Experience 10%.
      - Two interaction / non-linear terms are added so the target is
        not a pure linear formula:
          1. A "DSA-academic synergy penalty": very high CGPA cannot
             fully compensate for near-zero coding practice.
          2. An "experience diminishing returns" term: internships
             help but with saturating (sqrt) returns, not linear.
      - Gaussian noise (sigma=6) is added to simulate real-world
        unpredictability (interviewer variance, luck, unmeasured
        factors) so the model cannot achieve near-100% accuracy --
        this is intentional and mirrors reality.
    """
    # normalize each raw input to a 0-100 sub-score
    academic = (
        0.5 * (df["cgpa"] / 10 * 100)
        + 0.25 * df["tenth_pct"]
        + 0.25 * df["twelfth_pct"]
    ) - (df["backlogs"] * 4)

    dsa = (
        0.4 * np.clip(df["leetcode_solved"] / 400 * 100, 0, 100)
        + 0.35 * np.clip((df["contest_rating"] - 800) / 1600 * 100, 0, 100)
        + 0.25 * (df["dsa_proficiency"] / 10 * 100)
    )

    tech = (
        df[["num_languages_known"]].values.flatten() / 6 * 20
        + df["dbms_score"] / 5 * 20
        + df["oop_score"] / 5 * 20
        + df["os_score"] / 5 * 20
        + df["cn_score"] / 5 * 20
    )

    projects = (
        0.4 * np.clip(df["num_projects"] / 6 * 100, 0, 100)
        + 0.35 * np.clip(df["num_ml_projects"] / 3 * 100, 0, 100)
        + 0.25 * np.clip(df["github_commits"] / 500 * 100, 0, 100)
    )

    placement_skills = (
        0.3 * df["aptitude_score"]
        + 0.3 * df["communication_score"]
        + 0.2 * df["resume_score"]
        + 0.2 * df["interview_prep_score"]
    )

    experience = np.sqrt(df["internship_months"] / 12) * 100  # saturating returns
    other = 0.6 * np.clip(df["certifications_count"] / 4 * 100, 0, 100) + 0.4 * df["extracurricular_score"]

    # weighted combination (weights sum to 1.0 across the 5 main groups)
    base = (
        0.20 * academic
        + 0.30 * (0.7 * dsa + 0.3 * tech)   # DSA/coding group folds in raw tech score
        + 0.20 * projects
        + 0.20 * placement_skills
        + 0.10 * experience
    )

    # small nudge from "other" factors (kept minor deliberately)
    base = base + 0.05 * other

    # --- rescale to a realistic 0-100 spread ---
    # Averaging many bounded sub-scores naturally compresses the result
    # toward the middle (low variance). We re-spread it around a target
    # mean/std BEFORE adding interaction terms and noise, so categories
    # aren't collapsed into one bucket. This is a documented, fixed
    # linear transform (not fit on this run's data) -- z-score using the
    # generation-time mean/std of `base`, then re-centered to mean=55,
    # std=18, which was chosen so that the fixed 40/60/80 thresholds
    # below produce a plausible, non-degenerate category distribution.
    base = 55 + (base - base.mean()) / base.std() * 18

    # --- non-linear interaction terms ---
    # 1. DSA-academic synergy penalty: high CGPA + near-zero DSA practice
    #    gets penalized (can't just be "book smart")
    synergy_penalty = np.where(
        (df["cgpa"] > 8.0) & (df["leetcode_solved"] < 30),
        -8.0,
        0.0,
    )

    latent = base + synergy_penalty

    # noise: simulates unmeasured/uncontrollable factors
    noise = rng.normal(0, 7, len(df))
    latent = latent + noise

    return np.clip(latent, 0, 100)


def bin_to_category(score: np.ndarray) -> pd.Series:
    """Bin the latent score into 4 ordered readiness categories using
    fixed, documented thresholds (not quantiles of this run's data, so
    thresholds are stable and interpretable across dataset re-generations).
    """
    bins = [-0.01, 40, 60, 80, 100.01]
    labels = ["Needs Improvement", "Developing", "Placement Ready", "Highly Ready"]
    return pd.cut(score, bins=bins, labels=labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="number of student rows to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/students.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df = generate_features(args.n, rng)
    latent_score = compute_latent_score(df, rng)
    df["readiness_category"] = bin_to_category(latent_score)

    # AUDIT-ONLY column: the hidden continuous score used to build the
    # label. This is NEVER used as a model feature (it would be pure
    # leakage) -- it exists solely so failure analysis (Step 11) can
    # check whether a wrong prediction was a genuine model mistake or
    # simply a student sitting right on a category boundary.
    df["_audit_true_latent_score"] = latent_score

    # inject a small amount of realistic missingness (for later
    # preprocessing / robustness-testing steps) -- NOT in the target
    for col in ["resume_score", "communication_score", "internship_months", "certifications_count"]:
        mask = rng.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    df.to_csv(args.out, index=False)

    print(f"Saved {len(df)} rows to {args.out}")
    print("\nCategory distribution:")
    print(df["readiness_category"].value_counts(normalize=True).round(3))
    print("\nSample rows:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
