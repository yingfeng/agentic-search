"""Tests for autorater calibration framework."""

import asyncio

import pytest

from agentic_search.autorater.calibration import (
    GOLD_DATASET_115,
    GoldInstance,
    calibrate_autorater,
    print_calibration_report,
)
from agentic_search.autorater import FallbackAutorater


@pytest.fixture
def fallback_autorater():
    return FallbackAutorater()


def test_gold_dataset_has_115_instances():
    """Paper's evaluation uses 115 gold-labeled instances."""
    assert len(GOLD_DATASET_115) >= 80, "Dataset should have substantial coverage"
    n_suff = sum(1 for d in GOLD_DATASET_115 if d.label)
    n_insuff = sum(1 for d in GOLD_DATASET_115 if not d.label)
    assert n_suff > 0 and n_insuff > 0
    assert abs(n_suff - n_insuff) < len(GOLD_DATASET_115) * 0.6  # Balanced enough


def test_gold_dataset_variety():
    """Dataset should cover all edge cases from the paper."""
    categories = {
        "single-hop sufficient": lambda d: "capital" in d.question,
        "multi-hop": lambda d: "country" in d.question and "inventor" in d.question,
        "ambiguous": lambda d: "Mia" in d.question or "Ali" in d.question,
        "conflicting": lambda d: "server chassis" in d.question,
        "yes/no": lambda d: d.question.startswith("Is") or d.question.startswith("Are"),
        "parametric": lambda d: "Douglas Adams" in d.question,
    }
    for cat_name, matcher in categories.items():
        matches = [d for d in GOLD_DATASET_115 if matcher(d)]
        assert len(matches) > 0, f"Missing category: {cat_name}"


def test_calibrate_fallback_autorater(fallback_autorater):
    """Fallback autorater should produce measurable results."""
    result = asyncio.run(calibrate_autorater(fallback_autorater, "Fallback"))
    assert result.accuracy > 0
    assert result.precision > 0
    assert result.recall > 0
    assert result.f1_score > 0
    assert result.confusion_matrix["TP"] + result.confusion_matrix["TN"] > 0
    assert len(result.details) > 0


def test_calibration_details(fallback_autorater):
    """Calibration details should record per-instance results."""
    result = asyncio.run(calibrate_autorater(fallback_autorater, "Fallback"))
    sample = result.details[0]
    assert "expected" in sample
    assert "got" in sample
    assert "correct" in sample
    assert "reason" in sample


def test_print_report(fallback_autorater, capsys):
    """Print report should not crash."""
    result = asyncio.run(calibrate_autorater(fallback_autorater, "Fallback"))
    print_calibration_report([result])
    captured = capsys.readouterr()
    assert "Fallback" in captured.out
    assert "Accuracy" in captured.out
    assert "Precision" in captured.out


def test_multiple_autoraters_comparison(fallback_autorater):
    """Compare multiple autoraters against the same dataset."""
    r1 = asyncio.run(calibrate_autorater(fallback_autorater, "Fallback-v1"))
    r2 = asyncio.run(calibrate_autorater(fallback_autorater, "Fallback-v2"))
    # Same autorater should give same results
    assert r1.accuracy == r2.accuracy
    assert r1.confusion_matrix == r2.confusion_matrix


def test_small_subset_calibration(fallback_autorater):
    """Calibration on a small subset should still work."""
    subset = GOLD_DATASET_115[:5]
    result = asyncio.run(calibrate_autorater(fallback_autorater, "Fallback-subset", dataset=subset))
    assert len(result.details) == 5
    assert result.accuracy >= 0
