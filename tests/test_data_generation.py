"""Unit and integration tests for data generation and ingestion modules."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.generate_data import (
    CANONICAL_COLUMNS,
    generate_synthetic_data,
    ingest_real_data,
)


class TestSyntheticDataGeneration:
    """Test suite verifying PRD/SRS requirements for synthetic data synthesis."""

    def test_synthetic_shape_and_columns(self, tmp_path):
        """FR-1A: Verify record count and canonical columns."""
        out_file = tmp_path / "test_synthetic.csv"
        df = generate_synthetic_data(output_path=str(out_file), n=1000, seed=42)

        assert len(df) == 1000, "Synthetic dataset must contain exactly 1000 records."
        assert list(df.columns) == CANONICAL_COLUMNS, f"Columns must match canonical schema: {CANONICAL_COLUMNS}"
        assert out_file.exists(), "Output CSV file must be written to disk."

    def test_synthetic_reproducibility(self, tmp_path):
        """FR-1A: Verify seed 42 produces identical reproducible results."""
        out1 = tmp_path / "out1.csv"
        out2 = tmp_path / "out2.csv"

        df1 = generate_synthetic_data(output_path=str(out1), n=500, seed=42)
        df2 = generate_synthetic_data(output_path=str(out2), n=500, seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_synthetic_domain_constraints(self, tmp_path):
        """SRS §3.1: Verify all numerical boundary constraints are respected."""
        out_file = tmp_path / "test_constraints.csv"
        df = generate_synthetic_data(output_path=str(out_file), n=1000, seed=42)

        assert df["Age"].between(17, 25).all(), "Age must be within [17, 25]."
        assert df["Daily_Usage_Hours"].between(0.5, 9.8).all(), "Daily Usage must be within [0.5, 9.8]."
        assert df["Study_Hours"].between(0.5, 8.5).all(), "Study Hours must be within [0.5, 8.5]."
        assert df["Sleep_Hours"].between(3.5, 9.5).all(), "Sleep Hours must be within [3.5, 9.5]."
        assert df["Academic_Performance"].between(35.0, 98.5).all(), "Performance must be within [35.0, 98.5]."

    def test_synthetic_correlation_acceptance_criteria(self, tmp_path):
        """FR-3 / Acceptance Criteria: Verify usage vs sleep Pearson r <= -0.40."""
        out_file = tmp_path / "test_corr.csv"
        df = generate_synthetic_data(output_path=str(out_file), n=1000, seed=42)

        r_sleep = df["Daily_Usage_Hours"].corr(df["Sleep_Hours"])
        r_study = df["Daily_Usage_Hours"].corr(df["Study_Hours"])
        r_perf = df["Daily_Usage_Hours"].corr(df["Academic_Performance"])

        assert r_sleep <= -0.40, f"Usage vs Sleep correlation must satisfy r <= -0.40 (got {r_sleep:.4f})."
        assert r_study < 0, f"Usage vs Study correlation must be negative (got {r_study:.4f})."
        assert r_perf < 0, f"Usage vs Academic Performance must be negative (got {r_perf:.4f})."

    def test_synthetic_categories(self, tmp_path):
        """Verify categorical domain values for Gender and Platform."""
        out_file = tmp_path / "test_cat.csv"
        df = generate_synthetic_data(output_path=str(out_file), n=1000, seed=42)

        expected_genders = {"Male", "Female", "Other"}
        assert set(df["Gender"].unique()).issubset(expected_genders)

        expected_platforms = {"Instagram", "YouTube", "WhatsApp", "Snapchat", "Reddit", "LinkedIn"}
        assert set(df["Platform"].unique()).issubset(expected_platforms)
        assert df.isnull().sum().sum() == 0, "No null cells are permitted."


class TestRealDataIngestion:
    """Test suite verifying empirical survey ingestion and canonical mapping."""

    def test_real_data_ingestion(self, tmp_path):
        """FR-1B: Verify empirical dataset normalization into canonical schema."""
        out_file = tmp_path / "test_real.csv"
        candidate_paths = [
            Path("data/Student Social Media And Mental Health Impact.csv"),
            Path("dataset/Student Social Media And Mental Health Impact.csv"),
        ]
        input_csv = next((p for p in candidate_paths if p.exists()), None)
        assert input_csv is not None, "Empirical survey dataset must exist in data/ or dataset/."

        df = ingest_real_data(input_path=str(input_csv), output_path=str(out_file))

        assert len(df) == 5000, "Empirical dataset should contain 5000 cleaned records."
        assert list(df.columns) == CANONICAL_COLUMNS, "Normalized dataset must have canonical schema."
        assert df.isnull().sum().sum() == 0, "Cleaned dataset must have zero null values."
        assert df["Academic_Performance"].between(35.0, 98.5).all(), "Academic Performance must be bounded."

    def test_missing_file_raises_error(self, tmp_path):
        """Verify FileNotFoundError when input file does not exist."""
        with pytest.raises(FileNotFoundError):
            ingest_real_data(input_path="non_existent_path.csv", output_path=str(tmp_path / "out.csv"))
