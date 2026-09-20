"""Unit and integration tests for EDA analysis and visualization exporter."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.analyze import (
    display_eda_summary,
    generate_visualizations,
    load_dataset,
    run_analysis,
)
from src.generate_data import generate_synthetic_data


@pytest.fixture
def sample_canonical_df(tmp_path):
    """Provide a validated canonical dataframe for testing."""
    csv_path = tmp_path / "sample_data.csv"
    df = generate_synthetic_data(output_path=str(csv_path), n=200, seed=42)
    return df, csv_path


class TestAnalysisEngine:
    """Test suite for statistical analysis and visualization."""

    def test_load_dataset(self, sample_canonical_df):
        """Verify successful dataset loading."""
        _, csv_path = sample_canonical_df
        df = load_dataset(str(csv_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200

    def test_load_dataset_missing_raises(self):
        """Verify FileNotFoundError on missing dataset."""
        with pytest.raises(FileNotFoundError):
            load_dataset("non_existent_file.csv")

    def test_display_eda_summary(self, sample_canonical_df):
        """FR-2 & FR-3: Verify summary statistics and correlation matrix output."""
        df, _ = sample_canonical_df
        summary = display_eda_summary(df)

        assert "describe" in summary
        assert "correlation_matrix" in summary
        assert "platform_summary" in summary
        assert "gender_summary" in summary

        # 4x4 Pearson correlation matrix checks
        corr = summary["correlation_matrix"]
        assert corr.shape == (4, 4)
        for col in corr.columns:
            assert np.isclose(corr.loc[col, col], 1.0)
        # Symmetry check
        assert np.isclose(
            corr.loc["Daily_Usage_Hours", "Sleep_Hours"],
            corr.loc["Sleep_Hours", "Daily_Usage_Hours"],
        )

    def test_visualization_generation_and_dpi(self, sample_canonical_df, tmp_path):
        """FR-4 & NFR-3: Verify generation of 6 distinct 300 DPI figures."""
        df, _ = sample_canonical_df
        charts_dir = tmp_path / "charts"

        charts = generate_visualizations(df, charts_dir=str(charts_dir))

        expected_filenames = [
            "chart1_platform_usage.png",
            "chart2_daily_usage_hist.png",
            "chart3_age_group_usage.png",
            "chart4_study_vs_usage.png",
            "chart5_sleep_vs_usage.png",
            "chart6_usage_vs_performance.png",
        ]

        assert len(charts) == 6, "Must generate exactly 6 figures."

        for fname in expected_filenames:
            fig_path = charts_dir / fname
            assert fig_path.exists(), f"Figure {fname} must be created on disk."
            assert fig_path.stat().st_size > 5000, f"Figure {fname} must not be empty."

            # Verify 300 DPI resolution metadata
            with Image.open(fig_path) as img:
                dpi = img.info.get("dpi", (72, 72))
                # Matplotlib writes round(dpi) ~ 300
                assert dpi[0] >= 299, f"Figure {fname} DPI should be 300 (got {dpi[0]})."

    def test_run_analysis_end_to_end(self, sample_canonical_df, tmp_path):
        """Verify full run_analysis workflow execution."""
        _, csv_path = sample_canonical_df
        charts_dir = tmp_path / "charts_e2e"
        run_analysis(data_path=str(csv_path), charts_dir=str(charts_dir))
        assert len(list(charts_dir.glob("*.png"))) == 6
