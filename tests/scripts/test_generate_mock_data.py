"""Tests for deterministic mock-data generation."""

from pathlib import Path

from scripts.generate_mock_data import generate_mock_data


def test_generate_mock_data_creates_expected_files(tmp_path: Path) -> None:
    """Write the expected mock data artifacts to disk."""
    output_dir = tmp_path / "mock"
    generate_mock_data(output_dir=output_dir, seed=123, scale=10)
    assert (output_dir / "market_intelligence.csv").exists()
    assert (output_dir / "crm_notes.csv").exists()
    assert (output_dir / "product_kb.md").exists()
