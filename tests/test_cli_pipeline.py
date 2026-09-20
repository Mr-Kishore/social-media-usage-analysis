"""Integration tests for CLI orchestrator run.py and non-functional requirements."""

import subprocess
import sys
import time
from pathlib import Path
import pytest


class TestCLIPipeline:
    """Test suite verifying end-to-end execution and performance KPIs."""

    def test_run_synthetic_cli_and_latency(self):
        """FR-5 & NFR-1: Verify CLI synthetic execution completes in < 5.0s with exit code 0."""
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "run.py", "--data", "synthetic"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        elapsed = time.perf_counter() - start

        assert result.returncode == 0, f"run.py --data synthetic failed:\n{result.stderr}"
        assert elapsed < 5.0, f"Execution latency {elapsed:.2f}s exceeded 5.0s benchmark."
        assert "PASSED" in result.stdout

    def test_run_real_cli(self):
        """FR-5: Verify CLI real dataset ingestion completes with exit code 0."""
        result = subprocess.run(
            [sys.executable, "run.py", "--data", "real"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"run.py --data real failed:\n{result.stderr}"
        assert "Successfully normalized 5000 records" in result.stdout

    def test_cli_default_invocation(self):
        """Verify default execution (no arguments) completes with exit code 0."""
        result = subprocess.run(
            [sys.executable, "run.py"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"Default run.py invocation failed:\n{result.stderr}"
