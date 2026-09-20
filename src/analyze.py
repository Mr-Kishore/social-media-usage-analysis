"""Automated Exploratory Data Analysis and Publication-Quality Visualization Engine.

Generates:
1. Summary descriptive statistics table (console output).
2. Platform & Gender categorical distribution frequencies.
3. 4x4 Pearson correlation matrix.
4. Six publication-ready analytical figures saved at 300 DPI in charts/.
"""

import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_dataset(file_path: str = "data/social_media_usage_dataset.csv") -> pd.DataFrame:
    """Load canonical dataset from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path.resolve()}. Please run generate_data.py first."
        )
    df = pd.read_csv(path)
    return df


def display_eda_summary(df: pd.DataFrame) -> dict:
    """Compute and display descriptive statistics, category distributions, and correlation matrix.

    Args:
        df: Normalized canonical DataFrame.

    Returns:
        Dictionary containing summary tables and correlation matrix.
    """
    print("=" * 80)
    print("                EXPLORATORY DATA ANALYSIS (EDA) REPORT                ")
    print("=" * 80)

    print(f"\n[1] DATASET OVERVIEW: Total Records = {len(df)}, Features = {len(df.columns)}")
    print("-" * 80)
    print(df.dtypes.to_string())

    print("\n[2] DESCRIPTIVE SUMMARY STATISTICS:")
    print("-" * 80)
    num_cols = ["Daily_Usage_Hours", "Study_Hours", "Sleep_Hours", "Academic_Performance"]
    describe_df = df[num_cols].describe().round(3)
    print(describe_df.to_string())

    print("\n[3] CATEGORICAL DISTRIBUTIONS:")
    print("-" * 80)
    print("--- Preferred Social Media Platforms ---")
    plat_counts = df["Platform"].value_counts()
    plat_pct = df["Platform"].value_counts(normalize=True) * 100
    plat_summary = pd.DataFrame({"Count": plat_counts, "Percentage (%)": plat_pct.round(2)})
    print(plat_summary.to_string())

    print("\n--- Gender Representation ---")
    gender_counts = df["Gender"].value_counts()
    gender_pct = df["Gender"].value_counts(normalize=True) * 100
    gender_summary = pd.DataFrame({"Count": gender_counts, "Percentage (%)": gender_pct.round(2)})
    print(gender_summary.to_string())

    print("\n[4] MULTIVARIATE PEARSON CORRELATION MATRIX (4x4):")
    print("-" * 80)
    corr_matrix = df[num_cols].corr(method="pearson").round(4)
    print(corr_matrix.to_string())

    r_usage_sleep = corr_matrix.loc["Daily_Usage_Hours", "Sleep_Hours"]
    r_usage_study = corr_matrix.loc["Daily_Usage_Hours", "Study_Hours"]
    r_usage_perf = corr_matrix.loc["Daily_Usage_Hours", "Academic_Performance"]

    print("\n[5] STATISTICAL VALIDATION CHECKS:")
    print("-" * 80)
    print(f"* Daily Usage vs Sleep Hours:      r = {r_usage_sleep:+.4f}")
    print(f"* Daily Usage vs Study Hours:      r = {r_usage_study:+.4f}")
    print(f"* Daily Usage vs Performance:      r = {r_usage_perf:+.4f}")

    if r_usage_sleep <= -0.40:
        print("  -> ACCEPTANCE CRITERIA PASSED: Strong inverse usage-sleep relationship verified (r <= -0.40).")
    else:
        print("  -> ACCEPTANCE CRITERIA WARNING: Inverse correlation weaker than -0.40.")

    print("=" * 80)

    return {
        "describe": describe_df,
        "platform_summary": plat_summary,
        "gender_summary": gender_summary,
        "correlation_matrix": corr_matrix,
    }


def generate_visualizations(df: pd.DataFrame, charts_dir: str = "charts") -> list[str]:
    """Render and export six 300 DPI analytical charts.

    Figures generated:
    1. chart1_platform_usage.png: Descending platform countplot
    2. chart2_daily_usage_hist.png: Daily usage histogram + KDE (16 bins)
    3. chart3_age_group_usage.png: Mean daily usage across age groups (<18, 18-21, 22+)
    4. chart4_study_vs_usage.png: Linear regression plot of Study vs Usage
    5. chart5_sleep_vs_usage.png: Linear regression plot of Sleep vs Usage
    6. chart6_usage_vs_performance.png: Linear regression plot of Academic Performance vs Usage

    Args:
        df: Canonical DataFrame.
        charts_dir: Directory where figures are saved.

    Returns:
        List of generated file paths.
    """
    out_dir = Path(charts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_charts = []

    # Configure global styling
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
    })

    # -------------------------------------------------------------------------
    # Figure 1: Platform Countplot (Descending)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    order = df["Platform"].value_counts().index
    palette = sns.color_palette("mako", len(order))
    ax1 = sns.countplot(data=df, x="Platform", order=order, palette=palette, hue="Platform", legend=False)
    plt.title("Figure 1: Preferred Student Social Media Platforms", weight="bold", pad=15)
    plt.xlabel("Social Media Platform", weight="bold")
    plt.ylabel("Number of Students", weight="bold")
    plt.xticks(rotation=25, ha="right")

    # Add count labels on bars
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )
    fig1_path = out_dir / "chart1_platform_usage.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(str(fig1_path))

    # -------------------------------------------------------------------------
    # Figure 2: Daily Social Media Usage Distribution (16 bins + KDE)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    sns.histplot(
        df["Daily_Usage_Hours"],
        bins=16,
        kde=True,
        color="#2b5c8f",
        edgecolor="white",
        line_kws={"linewidth": 2.5, "color": "#e74c3c"},
    )
    mean_val = df["Daily_Usage_Hours"].mean()
    median_val = df["Daily_Usage_Hours"].median()
    plt.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}h")
    plt.axvline(median_val, color="#27ae60", linestyle=":", linewidth=2, label=f"Median: {median_val:.2f}h")
    plt.title("Figure 2: Distribution of Daily Social Media Usage", weight="bold", pad=15)
    plt.xlabel("Daily Usage (Hours)", weight="bold")
    plt.ylabel("Student Frequency", weight="bold")
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    fig2_path = out_dir / "chart2_daily_usage_hist.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(str(fig2_path))

    # -------------------------------------------------------------------------
    # Figure 3: Mean Daily Usage across Demographic Age Groups (<18, 18-21, 22+)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    age_bins = [-np.inf, 17.99, 21.0, np.inf]
    age_labels = ["<18", "18-21", "22+"]
    df_copy = df.copy()
    df_copy["Age_Group"] = pd.cut(df_copy["Age"], bins=age_bins, labels=age_labels)

    age_stats = (
        df_copy.groupby("Age_Group", observed=False)["Daily_Usage_Hours"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    palette3 = ["#4a90e2", "#50e3c2", "#b8e986"]
    ax3 = sns.barplot(
        data=age_stats,
        x="Age_Group",
        y="mean",
        palette=palette3,
        hue="Age_Group",
        legend=False,
        edgecolor="gray",
    )
    plt.title("Figure 3: Mean Daily Usage Hours across Age Cohorts", weight="bold", pad=15)
    plt.xlabel("Age Cohort", weight="bold")
    plt.ylabel("Mean Daily Usage (Hours)", weight="bold")
    plt.ylim(0, max(age_stats["mean"].fillna(0).max() * 1.25, 6))

    for idx, row in age_stats.iterrows():
        val = row["mean"]
        cnt = row["count"]
        if not np.isnan(val) and cnt > 0:
            ax3.annotate(
                f"{val:.2f} hrs\n(n={int(cnt)})",
                (idx, val),
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
                xytext=(0, 4),
                textcoords="offset points",
            )
    fig3_path = out_dir / "chart3_age_group_usage.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(str(fig3_path))

    # -------------------------------------------------------------------------
    # Figure 4: Study Hours vs. Daily Social Media Usage (Linear Regression)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    r_val4 = df["Daily_Usage_Hours"].corr(df["Study_Hours"])
    sns.regplot(
        data=df,
        x="Daily_Usage_Hours",
        y="Study_Hours",
        scatter_kws={"alpha": 0.25, "color": "#1f77b4", "s": 25},
        line_kws={"color": "#d62728", "linewidth": 2.5, "label": f"Fit Line (r = {r_val4:.2f})"},
    )
    plt.title("Figure 4: Academic Study Hours vs. Daily Social Media Usage", weight="bold", pad=15)
    plt.xlabel("Daily Social Media Usage (Hours)", weight="bold")
    plt.ylabel("Study Duration (Hours)", weight="bold")
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    fig4_path = out_dir / "chart4_study_vs_usage.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(str(fig4_path))

    # -------------------------------------------------------------------------
    # Figure 5: Sleep Hours vs. Daily Social Media Usage (Linear Regression)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    r_val5 = df["Daily_Usage_Hours"].corr(df["Sleep_Hours"])
    sns.regplot(
        data=df,
        x="Daily_Usage_Hours",
        y="Sleep_Hours",
        scatter_kws={"alpha": 0.25, "color": "#2ca02c", "s": 25},
        line_kws={"color": "#e377c2", "linewidth": 2.5, "label": f"Fit Line (r = {r_val5:.2f})"},
    )
    plt.title("Figure 5: Sleep Duration vs. Daily Social Media Usage", weight="bold", pad=15)
    plt.xlabel("Daily Social Media Usage (Hours)", weight="bold")
    plt.ylabel("Sleep Duration (Hours)", weight="bold")
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    fig5_path = out_dir / "chart5_sleep_vs_usage.png"
    plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(str(fig5_path))

    # -------------------------------------------------------------------------
    # Figure 6: Academic Performance vs. Daily Usage (Linear Regression)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    r_val6 = df["Daily_Usage_Hours"].corr(df["Academic_Performance"])
    sns.regplot(
        data=df,
        x="Daily_Usage_Hours",
        y="Academic_Performance",
        scatter_kws={"alpha": 0.25, "color": "#9467bd", "s": 25},
        line_kws={"color": "#ff7f0e", "linewidth": 2.5, "label": f"Fit Line (r = {r_val6:.2f})"},
    )
    plt.title("Figure 6: Academic Exam Performance vs. Daily Usage", weight="bold", pad=15)
    plt.xlabel("Daily Social Media Usage (Hours)", weight="bold")
    plt.ylabel("Academic Performance (%)", weight="bold")
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    fig6_path = out_dir / "chart6_usage_vs_performance.png"
    plt.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(str(fig6_path))

    print(f"\n[Export Engine] Successfully generated {len(generated_charts)} publication-ready figures (300 DPI):")
    for cp in generated_charts:
        print(f"  * {cp}")

    return generated_charts


def run_analysis(
    data_path: str = "data/social_media_usage_dataset.csv",
    charts_dir: str = "charts",
) -> None:
    """Execute complete analysis pipeline: load, compute statistics, export charts."""
    df = load_dataset(data_path)
    display_eda_summary(df)
    generate_visualizations(df, charts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis & Chart Generation")
    parser.add_argument(
        "--data",
        default="data/social_media_usage_dataset.csv",
        help="Path to canonical CSV dataset",
    )
    parser.add_argument(
        "--charts",
        default="charts",
        help="Output directory for generated charts",
    )
    args = parser.parse_args()
    run_analysis(data_path=args.data, charts_dir=args.charts)
