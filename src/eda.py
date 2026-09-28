"""
eda.py
------
Exploratory Data Analysis for SkillLens.

Run:
    python src/eda.py

Produces:
    - printed summary statistics + correlations
    - reports/figures/*.png (category balance, score distributions,
      correlation heatmap, key scatter relationships)

This is the script version of notebooks/exploratory_analysis.ipynb --
run either one, they cover the same analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CATEGORY_ORDER = ["Needs Improvement", "Developing", "Placement Ready", "Highly Ready"]


def load_data(path="data/raw/students.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["readiness_category"] = pd.Categorical(
        df["readiness_category"], categories=CATEGORY_ORDER, ordered=True
    )
    return df


def print_overview(df: pd.DataFrame):
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("\nMissing values per column:")
    miss = df.isna().sum()
    print(miss[miss > 0])
    print("\nCategory distribution:")
    print(df["readiness_category"].value_counts(normalize=True).round(3).sort_index())
    print("\nNumeric summary (selected columns):")
    print(df[["cgpa", "leetcode_solved", "aptitude_score", "num_projects"]].describe().round(2))


def plot_category_balance(df: pd.DataFrame, out="reports/figures/01_category_balance.png"):
    plt.figure(figsize=(6, 4))
    order = CATEGORY_ORDER
    counts = df["readiness_category"].value_counts().reindex(order)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    plt.title("Readiness Category Distribution")
    plt.ylabel("Number of students")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


def plot_score_distributions(df: pd.DataFrame, out="reports/figures/02_key_score_distributions.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    cols = ["cgpa", "leetcode_solved", "aptitude_score", "num_projects"]
    for ax, col in zip(axes.flat, cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution: {col}")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


def plot_correlation_heatmap(df: pd.DataFrame, out="reports/figures/03_correlation_heatmap.png"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


def plot_feature_vs_target(df: pd.DataFrame, out="reports/figures/04_feature_vs_readiness.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    pairs = ["cgpa", "leetcode_solved", "num_projects", "aptitude_score"]
    for ax, col in zip(axes.flat, pairs):
        sns.boxplot(data=df, x="readiness_category", y=col, order=CATEGORY_ORDER, hue="readiness_category",
                    palette="viridis", ax=ax, legend=False)
        ax.set_title(f"{col} vs readiness_category")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


def main():
    df = load_data()
    print_overview(df)
    plot_category_balance(df)
    plot_score_distributions(df)
    plot_correlation_heatmap(df)
    plot_feature_vs_target(df)

    print("\nTop correlations with numeric target proxy (aptitude/cgpa/leetcode combined rank):")
    # quick sanity check: does an increasing ordinal category align with feature medians?
    for col in ["cgpa", "leetcode_solved", "aptitude_score", "num_projects", "communication_score"]:
        medians = df.groupby("readiness_category", observed=True)[col].median()
        print(f"  {col}: {medians.to_dict()}")


if __name__ == "__main__":
    main()
