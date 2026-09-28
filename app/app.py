"""
app.py
------
SkillLens Streamlit app.

Tabs:
  1. Single-Student Prediction & What-If (Step 8)
  2. CSV Batch Prediction (Step 9)

The dashboard (Step 10) is added as a third tab in a later pass --
this file grows, it is not being rewritten from scratch each time.

Run:
    streamlit run app/app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from predict import load_artifacts, predict_from_raw, RAW_FEATURE_COLS, CATEGORY_ORDER
from explain import explain_one_student_score
from csv_validation import validate_and_clean, CSVValidationError
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="SkillLens", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def get_defaults():
    """Median of each raw feature from the training data -- used as
    sensible slider starting points, and as the neutral baseline the
    explanation feature-ablation compares against.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "students.csv"))
    return df[RAW_FEATURE_COLS].median().to_dict()


@st.cache_data
def get_scored_dataset():
    """Score the full raw dataset with the final model, for the
    dashboard. In a real deployment this would be the institution's
    current student roster (e.g. uploaded once via the CSV tab); here
    we use the same students.csv the model was trained/tested on, so
    the dashboard has something realistic to show out of the box.
    """
    base_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base_dir, "..", "data", "raw", "students.csv"))
    df_features = df[RAW_FEATURE_COLS].copy()
    # median-fill for scoring purposes only (pipeline also imputes internally,
    # this just keeps the raw display columns clean of NaN for charting)
    df_features = df_features.fillna(df_features.median(numeric_only=True))
    scored = predict_from_raw(df_features, pipeline, model)
    return scored


@st.cache_data
def get_global_importance():
    base_dir = os.path.dirname(__file__)
    X_test = pd.read_csv(os.path.join(base_dir, "..", "data", "processed", "X_test.csv"))
    y_test = pd.read_csv(os.path.join(base_dir, "..", "data", "processed", "y_test.csv"))["readiness_category"]
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1_macro", n_jobs=-1
    )
    names = [c.split("__")[-1] for c in X_test.columns]
    importance = pd.Series(result.importances_mean, index=names).sort_values(ascending=False)
    return importance


pipeline, model = get_artifacts()
defaults = get_defaults()

st.title("SkillLens")
st.caption("AI-powered Placement Readiness Analysis")
st.info(
    "This is a model simulation trained on synthetic data. Scores and categories "
    "reflect patterns the model learned -- they are not a guarantee of actual "
    "placement outcomes."
)

tab_single, tab_csv, tab_dashboard = st.tabs(
    ["Single-Student Prediction & What-If", "CSV Batch Prediction", "Dashboard"]
)


# ============================================================
# TAB 1: Single-student prediction + what-if simulator
# ============================================================
with tab_single:
    col_academic, col_coding, col_tech = st.columns(3)

    with col_academic:
        st.subheader("Academics")
        cgpa = st.slider("CGPA", 4.0, 10.0, float(defaults["cgpa"]), 0.1)
        tenth_pct = st.slider("10th percentage", 40.0, 100.0, float(defaults["tenth_pct"]), 0.5)
        twelfth_pct = st.slider("12th percentage", 40.0, 100.0, float(defaults["twelfth_pct"]), 0.5)
        backlogs = st.number_input("Active backlogs", 0, 10, int(defaults["backlogs"]))

    with col_coding:
        st.subheader("Coding / DSA")
        leetcode_solved = st.slider("LeetCode problems solved", 0, 800, int(defaults["leetcode_solved"]), 5)
        contest_rating = st.slider("Coding contest rating", 800, 2400, int(defaults["contest_rating"]), 10)
        dsa_proficiency = st.slider("Self-rated DSA proficiency (1-10)", 1, 10, int(defaults["dsa_proficiency"]))
        num_languages_known = st.slider("Programming languages known", 1, 6, int(defaults["num_languages_known"]))

    with col_tech:
        st.subheader("Core CS knowledge (1-5)")
        dbms_score = st.slider("DBMS", 1, 5, int(defaults["dbms_score"]))
        oop_score = st.slider("OOP", 1, 5, int(defaults["oop_score"]))
        os_score = st.slider("Operating Systems", 1, 5, int(defaults["os_score"]))
        cn_score = st.slider("Computer Networks", 1, 5, int(defaults["cn_score"]))

    col_projects, col_exp, col_skills = st.columns(3)

    with col_projects:
        st.subheader("Projects")
        num_projects = st.slider("Number of projects", 0, 10, int(defaults["num_projects"]))
        num_ml_projects = st.slider("Number of ML projects", 0, 5, int(defaults["num_ml_projects"]))
        github_commits = st.slider("GitHub commits (total)", 0, 2000, int(defaults["github_commits"]), 25)

    with col_exp:
        st.subheader("Experience")
        has_internship = st.selectbox("Has completed an internship?", ["No", "Yes"],
                                       index=int(defaults["has_internship"]))
        internship_months = st.slider("Internship duration (months)", 0.0, 12.0,
                                       float(defaults["internship_months"]), 0.5,
                                       disabled=(has_internship == "No"))

    with col_skills:
        st.subheader("Placement skills (0-100)")
        aptitude_score = st.slider("Aptitude score", 0.0, 100.0, float(defaults["aptitude_score"]), 1.0)
        communication_score = st.slider("Communication score", 0.0, 100.0, float(defaults["communication_score"]), 1.0)
        resume_score = st.slider("Resume quality score", 0.0, 100.0, float(defaults["resume_score"]), 1.0)
        interview_prep_score = st.slider("Interview prep score", 0.0, 100.0, float(defaults["interview_prep_score"]), 1.0)

    st.subheader("Other")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        certifications_count = st.slider("Certifications completed", 0, 10, int(defaults["certifications_count"]))
    with col_o2:
        extracurricular_score = st.slider("Extracurricular involvement score", 0.0, 100.0,
                                           float(defaults["extracurricular_score"]), 1.0)

    current_raw = {
        "cgpa": cgpa, "tenth_pct": tenth_pct, "twelfth_pct": twelfth_pct, "backlogs": backlogs,
        "leetcode_solved": leetcode_solved, "contest_rating": contest_rating,
        "dsa_proficiency": dsa_proficiency, "num_languages_known": num_languages_known,
        "dbms_score": dbms_score, "oop_score": oop_score, "os_score": os_score, "cn_score": cn_score,
        "num_projects": num_projects, "num_ml_projects": num_ml_projects, "github_commits": github_commits,
        "has_internship": 1 if has_internship == "Yes" else 0,
        "internship_months": internship_months if has_internship == "Yes" else 0.0,
        "aptitude_score": aptitude_score, "communication_score": communication_score,
        "resume_score": resume_score, "interview_prep_score": interview_prep_score,
        "certifications_count": certifications_count, "extracurricular_score": extracurricular_score,
    }

    raw_df = pd.DataFrame([current_raw])
    result = predict_from_raw(raw_df, pipeline, model)
    score = float(result["readiness_score"].iloc[0])
    category = result["predicted_category"].iloc[0]

    st.divider()
    result_col, factor_col = st.columns([1, 2])

    with result_col:
        st.metric("Readiness Score", f"{score:.1f} / 100")
        st.metric("Readiness Category", category)

        if "prev_raw" in st.session_state and "prev_score" in st.session_state:
            delta = score - st.session_state.prev_score
            st.metric("Change since last edit", f"{delta:+.1f} points")

    with factor_col:
        st.markdown("**Top factors for this student** (vs. an average student in the training data)")
        ranked, _ = explain_one_student_score(raw_df, pipeline, model)
        factor_df = pd.DataFrame(ranked, columns=["Feature", "Contribution (points)"]).head(6)
        factor_df["Direction"] = factor_df["Contribution (points)"].apply(
            lambda x: "Increases readiness" if x > 0 else "Decreases readiness"
        )
        factor_df["Contribution (points)"] = factor_df["Contribution (points)"].round(2)
        st.dataframe(factor_df, hide_index=True, use_container_width=True)

        if "prev_raw" in st.session_state:
            changed_features = [f for f in RAW_FEATURE_COLS if st.session_state.prev_raw[f] != current_raw[f]]
            if changed_features:
                st.markdown("**What changed since your last edit:**")
                change_rows = []
                for f in changed_features:
                    reverted = current_raw.copy()
                    reverted[f] = st.session_state.prev_raw[f]
                    reverted_score = predict_from_raw(pd.DataFrame([reverted]), pipeline, model)["readiness_score"].iloc[0]
                    change_rows.append({
                        "Feature": f,
                        "From": st.session_state.prev_raw[f],
                        "To": current_raw[f],
                        "Points contributed": round(score - reverted_score, 2),
                    })
                st.dataframe(pd.DataFrame(change_rows), hide_index=True, use_container_width=True)

    st.caption(
        "This what-if tool is a model simulation only. Changing these values does not "
        "change any real academic record and does not guarantee a real placement outcome."
    )

    st.session_state.prev_raw = current_raw
    st.session_state.prev_score = score


# ============================================================
# TAB 2: CSV batch prediction (Step 9)
# ============================================================
with tab_csv:
    st.subheader("CSV Batch Prediction")
    st.markdown(
        "Upload a CSV with one row per student. Required columns:\n\n"
        f"`{', '.join(RAW_FEATURE_COLS)}`\n\n"
        "Extra columns are kept and passed through untouched. Missing values in "
        "required numeric columns are automatically filled using the same "
        "median-imputation the model was trained with."
    )

    uploaded = st.file_uploader("Upload student data (.csv)", type=["csv"])

    if uploaded is not None:
        # --- the file itself might not even parse ---
        try:
            df_upload = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read this file as a CSV: {e}")
            st.stop()

        # --- validate + clean (raises CSVValidationError for empty file
        #     or missing required columns; repairs everything else) ---
        try:
            working, extra_cols, coercion_report, total_missing = validate_and_clean(df_upload)
        except CSVValidationError as e:
            st.error(str(e))
            st.stop()

        if extra_cols:
            st.warning(f"Ignoring {len(extra_cols)} extra column(s) not used by the model: {', '.join(extra_cols)}")

        if coercion_report:
            st.warning(
                "Some values could not be interpreted as numbers and were treated as missing "
                "(then median-imputed): " +
                ", ".join(f"{col} ({n} row(s))" for col, n in coercion_report.items())
            )

        if total_missing > 0:
            st.info(f"{total_missing} missing value(s) across all columns will be median-imputed by the pipeline.")

        # --- run predictions ---
        try:
            predictions = predict_from_raw(working, pipeline, model)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.success(f"Generated predictions for {len(predictions)} student(s).")
        display_cols = list(df_upload.columns) + ["predicted_category", "readiness_score"] + \
            [c for c in predictions.columns if c.startswith("prob_")]
        st.dataframe(predictions[display_cols], use_container_width=True)

        csv_bytes = predictions[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions as CSV",
            data=csv_bytes,
            file_name="skilllens_predictions.csv",
            mime="text/csv",
        )
    else:
        st.caption("No file uploaded yet.")


# ============================================================
# TAB 3: Dashboard (Step 10)
# ============================================================
with tab_dashboard:
    st.subheader("Dashboard")

    scored = get_scored_dataset()

    st.markdown("**Filters**")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        category_filter = st.multiselect("Readiness category", CATEGORY_ORDER, default=CATEGORY_ORDER)
    with f2:
        cgpa_min, cgpa_max = float(scored["cgpa"].min()), float(scored["cgpa"].max())
        cgpa_range = st.slider("CGPA range", cgpa_min, cgpa_max, (cgpa_min, cgpa_max))
    with f3:
        dsa_min, dsa_max = int(scored["leetcode_solved"].min()), int(scored["leetcode_solved"].max())
        dsa_range = st.slider("LeetCode solved range", dsa_min, dsa_max, (dsa_min, dsa_max))
    with f4:
        internship_filter = st.selectbox("Internship status", ["All", "Has internship", "No internship"])

    filtered = scored[
        scored["predicted_category"].isin(category_filter)
        & scored["cgpa"].between(cgpa_range[0], cgpa_range[1])
        & scored["leetcode_solved"].between(dsa_range[0], dsa_range[1])
    ]
    if internship_filter == "Has internship":
        filtered = filtered[filtered["has_internship"] == 1]
    elif internship_filter == "No internship":
        filtered = filtered[filtered["has_internship"] == 0]

    if filtered.empty:
        st.warning("No students match the current filters.")
        st.stop()

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students analyzed", len(filtered))
    c2.metric("Avg readiness score", f"{filtered['readiness_score'].mean():.1f}")
    ready_pct = filtered["predicted_category"].isin(["Placement Ready", "Highly Ready"]).mean() * 100
    c3.metric("Placement-ready %", f"{ready_pct:.1f}%")
    c4.metric("Avg CGPA", f"{filtered['cgpa'].mean():.2f}")
    c5.metric("Avg DSA (LeetCode)", f"{filtered['leetcode_solved'].mean():.0f}")

    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Readiness category distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        counts = filtered["predicted_category"].value_counts().reindex(CATEGORY_ORDER).fillna(0)
        ax.bar(counts.index, counts.values, color="teal")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**CGPA vs. readiness category**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        data_by_cat = [filtered.loc[filtered["predicted_category"] == c, "cgpa"] for c in CATEGORY_ORDER]
        ax.boxplot(data_by_cat, tick_labels=CATEGORY_ORDER)
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Number of projects vs. readiness category**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        data_by_cat = [filtered.loc[filtered["predicted_category"] == c, "num_projects"] for c in CATEGORY_ORDER]
        ax.boxplot(data_by_cat, tick_labels=CATEGORY_ORDER)
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        st.markdown("**Readiness score distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(filtered["readiness_score"], bins=20, color="steelblue")
        ax.set_xlabel("Readiness score")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**LeetCode solved vs. readiness category**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        data_by_cat = [filtered.loc[filtered["predicted_category"] == c, "leetcode_solved"] for c in CATEGORY_ORDER]
        ax.boxplot(data_by_cat, tick_labels=CATEGORY_ORDER)
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Avg readiness score: internship vs. no internship**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        grp = filtered.groupby("has_internship")["readiness_score"].mean()
        labels = ["No internship" if i == 0 else "Has internship" for i in grp.index]
        ax.bar(labels, grp.values, color=["indianred", "seagreen"])
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("**Global feature importance** (permutation importance, macro-F1 drop, computed once on the held-out test set)")
    importance = get_global_importance()
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.head(12).sort_values().plot(kind="barh", color="teal", ax=ax)
    ax.set_xlabel("Mean decrease in macro-F1 when feature is shuffled")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "This dashboard scores the same synthetic dataset the model was trained and tested on, "
        "for demonstration. In real use it would score an uploaded student roster (see the CSV tab)."
    )
