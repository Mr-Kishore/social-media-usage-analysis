"""Unified CLI Orchestrator for the Social Media Usage Analysis System.

Undergraduate Project: Bachelor of Science in Computer Technology
Department of Computer Technology, Dr. N.G.P. Arts and Science College (Bharathiar University)

Supported Commands:
    python run.py --data synthetic   # Runs synthetic cohort generation (N=1000, seed=42) + EDA
    python run.py --data real        # Ingests empirical survey CSV (dataset/) + EDA
    python run.py                    # Defaults to synthetic mode with explicit status banner
"""

import argparse
import sys
import time
from pathlib import Path
from src.analyze import run_analysis
from src.generate_data import generate_synthetic_data, ingest_real_data


def print_banner(mode: str) -> None:
    """Print project header information."""
    print("=" * 80)
    print("           SOCIAL MEDIA USAGE ANALYSIS SYSTEM - B.Sc. COMPUTER TECHNOLOGY     ")
    print("                 Dr. N.G.P. Arts and Science College (Autonomous)             ")
    print("=" * 80)
    print(f"[*] Execution Mode: {mode.upper()}")
    print(f"[*] Standard Schema Target: data/social_media_usage_dataset.csv")
    print(f"[*] Visualization Output:   charts/ (300 DPI publication standards)")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified Pipeline Orchestrator for Student Social Media Usage Analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Select data source:\n"
             "  'synthetic': Generate parameterized reproducible cohort (N=1,000, seed=42)\n"
             "  'real'     : Ingest & harmonize empirical survey dataset (dataset/)",
    )
    parser.add_argument(
        "--input",
        default="dataset/Student Social Media And Mental Health Impact.csv",
        help="Input CSV file path for real mode (default: dataset/Student Social Media And Mental Health Impact.csv)",
    )
    parser.add_argument(
        "--output-data",
        default="data/social_media_usage_dataset.csv",
        help="Path for canonical output dataset (default: data/social_media_usage_dataset.csv)",
    )
    parser.add_argument(
        "--charts-dir",
        default="charts",
        help="Output directory for generated figures (default: charts/)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of records to synthesize (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic PRNG seed for synthetic data (default: 42)",
    )

    args = parser.parse_args()
    start_time = time.perf_counter()

    print_banner(args.data)

    try:
        # Step 1: Data Preparation / Ingestion
        print(f"\n>>> PHASE 1: DATA PREPARATION & NORMALIZATION ({args.data.upper()} MODE)...")
        if args.data == "synthetic":
            generate_synthetic_data(
                output_path=args.output_data,
                n=args.n,
                seed=args.seed,
            )
        elif args.data == "real":
            ingest_real_data(
                input_path=args.input,
                output_path=args.output_data,
            )

        # Step 2: Exploratory Data Analysis & Visualization
        print(f"\n>>> PHASE 2: EXPLORATORY DATA ANALYSIS & VISUALIZATION RENDERING...")
        run_analysis(
            data_path=args.output_data,
            charts_dir=args.charts_dir,
        )

        elapsed = time.perf_counter() - start_time
        print("\n" + "=" * 80)
        print(f"[SUCCESS] Pipeline completed in {elapsed:.2f} seconds.")
        print(f"[*] KPI Benchmark: {'PASSED (< 5.0s)' if elapsed <= 5.0 else 'EXCEEDED (> 5.0s)'}")
        print(f"[*] Canonical Dataset: {Path(args.output_data).resolve()}")
        print(f"[*] Visual Figures:    {Path(args.charts_dir).resolve()}")
        print("=" * 80)
        return 0

    except Exception as exc:
        print(f"\n[ERROR] Pipeline execution failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
