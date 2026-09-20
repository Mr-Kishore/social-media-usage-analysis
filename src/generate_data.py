"""Data preparation and ingestion module for Social Media Usage Analysis System.

Supports dual data modes:
1. Synthetic parameter-driven generation (reproducible seed, N=1000).
2. Empirical dataset ingestion and normalization into canonical 7-attribute schema.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Canonical entity attributes
CANONICAL_COLUMNS = [
    "Age",
    "Gender",
    "Platform",
    "Daily_Usage_Hours",
    "Study_Hours",
    "Sleep_Hours",
    "Academic_Performance",
]


def generate_synthetic_data(
    output_path: str = "data/social_media_usage_dataset.csv",
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic, statistically parameterized synthetic student cohort.

    Args:
        output_path: Path where canonical CSV will be saved.
        n: Cohort sample size (default 1000).
        seed: Random seed for exact reproducibility (default 42).

    Returns:
        pd.DataFrame containing the generated canonical dataset.
    """
    rng = np.random.default_rng(seed)

    # 1. Age: Discrete distribution centered on 19-21, covering [17, 25]
    age_choices = np.array([17, 18, 19, 20, 21, 22, 23, 24, 25])
    age_probs = np.array([0.05, 0.10, 0.22, 0.26, 0.20, 0.08, 0.05, 0.03, 0.01])
    age_probs = age_probs / age_probs.sum()
    ages = rng.choice(age_choices, size=n, p=age_probs)

    # 2. Gender: Multi-class categorical
    gender_choices = ["Male", "Female", "Other"]
    gender_probs = [0.49, 0.49, 0.02]
    genders = rng.choice(gender_choices, size=n, p=gender_probs)

    # 3. Platform: Defined by PRD empirical proportions
    # Instagram: 35%, YouTube: 25%, WhatsApp: 18%, Snapchat: 12%, Reddit: 6%, LinkedIn: 4%
    platform_choices = ["Instagram", "YouTube", "WhatsApp", "Snapchat", "Reddit", "LinkedIn"]
    platform_probs = [0.35, 0.25, 0.18, 0.12, 0.06, 0.04]
    platforms = rng.choice(platform_choices, size=n, p=platform_probs)

    # 4. Daily_Usage_Hours: N(4.2, 1.7^2), bounded within [0.5, 9.8]
    daily_usage = rng.normal(loc=4.2, scale=1.7, size=n)
    daily_usage = np.clip(daily_usage, 0.5, 9.8)

    # 5. Sleep_Hours: 8.2 - 0.28(Usage) + eps_sleep, eps_sleep ~ N(0, 0.45^2), bounded [3.5, 9.5]
    eps_sleep = rng.normal(loc=0.0, scale=0.45, size=n)
    sleep_hours = 8.2 - (0.28 * daily_usage) + eps_sleep
    sleep_hours = np.clip(sleep_hours, 3.5, 9.5)

    # 6. Study_Hours: 5.8 - 0.35(Usage) + eps_study, eps_study ~ N(0, 0.50^2), bounded [0.5, 8.5]
    eps_study = rng.normal(loc=0.0, scale=0.50, size=n)
    study_hours = 5.8 - (0.35 * daily_usage) + eps_study
    study_hours = np.clip(study_hours, 0.5, 8.5)

    # 7. Academic_Performance: 52.0 + 3.8(Study) + 1.6(Sleep) - 2.1(Usage) + eps_perf
    # eps_perf ~ N(0, 3.0^2), bounded within [35.0, 98.5]
    eps_perf = rng.normal(loc=0.0, scale=3.0, size=n)
    academic_perf = 52.0 + (3.8 * study_hours) + (1.6 * sleep_hours) - (2.1 * daily_usage) + eps_perf
    academic_perf = np.clip(academic_perf, 35.0, 98.5)

    df = pd.DataFrame({
        "Age": ages.astype(int),
        "Gender": genders,
        "Platform": platforms,
        "Daily_Usage_Hours": np.round(daily_usage, 2),
        "Study_Hours": np.round(study_hours, 2),
        "Sleep_Hours": np.round(sleep_hours, 2),
        "Academic_Performance": np.round(academic_perf, 1),
    })[CANONICAL_COLUMNS]

    # Save to CSV
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Correlation verification
    r_sleep = df["Daily_Usage_Hours"].corr(df["Sleep_Hours"])
    print(f"[Synthetic Generator] Generated {len(df)} records with seed {seed}.")
    print(f"[Synthetic Generator] Usage vs Sleep Pearson r: {r_sleep:.4f} (Criterion: r <= -0.40)")
    print(f"[Synthetic Generator] Dataset saved to: {output_path}")

    return df


def ingest_real_data(
    input_path: str = "dataset/Student Social Media And Mental Health Impact.csv",
    output_path: str = "data/social_media_usage_dataset.csv",
) -> pd.DataFrame:
    """Ingest, clean, and harmonize empirical survey data into canonical 7-attribute schema.

    Args:
        input_path: Path to raw survey CSV file.
        output_path: Destination path for normalized CSV dataset.

    Returns:
        pd.DataFrame containing normalized canonical records.
    """
    raw_path = Path(input_path)
    if not raw_path.exists():
        if raw_path.name == "Student Social Media And Mental Health Impact.csv":
            candidate_paths = [
                Path("data/Student Social Media And Mental Health Impact.csv"),
                Path("dataset/Student Social Media And Mental Health Impact.csv"),
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    raw_path = candidate
                    break
            else:
                raise FileNotFoundError(f"Empirical dataset not found at: {Path(input_path).resolve()}")
        else:
            raise FileNotFoundError(f"Empirical dataset not found at: {Path(input_path).resolve()}")

    raw_df = pd.read_csv(raw_path)
    print(f"[Real Data Ingestor] Ingested {len(raw_df)} raw records from {raw_path}.")

    # Schema Harmonization as per SRS §3.2
    # Check required columns
    required_raw = [
        "Age",
        "Gender",
        "Most_Used_Platform",
        "Avg_Daily_Usage_Hours",
        "Study_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
    ]
    missing = [c for c in required_raw if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")

    # Clean missing values
    clean_df = raw_df.dropna(subset=required_raw).copy()

    # Transformations
    ages = clean_df["Age"].astype(int).clip(lower=17, upper=25)
    genders = clean_df["Gender"].astype(str).str.strip()
    platforms = clean_df["Most_Used_Platform"].astype(str).str.strip()
    daily_usage = clean_df["Avg_Daily_Usage_Hours"].astype(float).clip(0.5, 9.8).round(2)
    study_hours = clean_df["Study_Hours"].astype(float).clip(0.5, 8.5).round(2)
    sleep_hours = clean_df["Sleep_Hours_Per_Night"].astype(float).clip(3.5, 9.5).round(2)

    # Academic_Performance derived from Mental_Health_Score * 10
    academic_perf = (clean_df["Mental_Health_Score"].astype(float) * 10.0).clip(35.0, 98.5).round(1)

    canonical_df = pd.DataFrame({
        "Age": ages,
        "Gender": genders,
        "Platform": platforms,
        "Daily_Usage_Hours": daily_usage,
        "Study_Hours": study_hours,
        "Sleep_Hours": sleep_hours,
        "Academic_Performance": academic_perf,
    })[CANONICAL_COLUMNS]

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_df.to_csv(output_path, index=False)

    print(f"[Real Data Ingestor] Successfully normalized {len(canonical_df)} records.")
    print(f"[Real Data Ingestor] Saved canonical dataset to: {output_path}")

    return canonical_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare or Ingest Student Social Media Dataset")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Data generation mode: 'synthetic' or 'real' (default: synthetic)",
    )
    parser.add_argument(
        "--input",
        default="dataset/Student Social Media And Mental Health Impact.csv",
        help="Input CSV path for real mode",
    )
    parser.add_argument(
        "--output",
        default="data/social_media_usage_dataset.csv",
        help="Output CSV path for normalized dataset",
    )
    parser.add_argument("--n", type=int, default=1000, help="Record count for synthetic mode")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic mode")

    args = parser.parse_args()

    if args.mode == "synthetic":
        generate_synthetic_data(output_path=args.output, n=args.n, seed=args.seed)
    else:
        ingest_real_data(input_path=args.input, output_path=args.output)
