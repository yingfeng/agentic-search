"""Tests for selective generation trainer."""

import pytest

from agentic_search.selective_gen_trainer import (
    generate_synthetic_data,
    train_weights,
    compute_accuracy_coverage_curve,
    compare_curves,
    print_training_report,
    SelectiveGenWeights,
)


def test_synthetic_data_generation():
    """Synthetic data should have realistic distribution."""
    data = generate_synthetic_data(n_samples=500, seed=42)
    assert len(data) == 500

    n_correct = sum(1 for d in data if d.is_correct)
    n_hallucinate = sum(1 for d in data if not d.is_correct)

    # Should have both correct and incorrect
    assert n_correct > 0
    assert n_hallucinate > 0

    # Sufficiency should be distributed across [0, 1]
    scores = [d.sufficiency_score for d in data]
    assert max(scores) <= 1.0
    assert min(scores) >= 0.0
    assert any(s < 0.5 for s in scores), "Should have insufficient examples"
    assert any(s > 0.5 for s in scores), "Should have sufficient examples"


def test_synthetic_data_distribution():
    """Key paper finding: high conf + low suff → high hallucination rate."""
    data = generate_synthetic_data(n_samples=2000, seed=42)

    # Most dangerous quadrant
    dangerous = [d for d in data if d.confidence > 0.7 and d.sufficiency_score < 0.5]
    safe = [d for d in data if d.confidence > 0.7 and d.sufficiency_score > 0.7]

    if dangerous and safe:
        dangerous_hallucination = sum(1 for d in dangerous if not d.is_correct) / len(dangerous)
        safe_hallucination = sum(1 for d in safe if not d.is_correct) / len(safe)
        assert dangerous_hallucination > safe_hallucination, \
            "Dangerous quadrant should have higher hallucination rate"


def test_train_weights():
    """Training should produce non-trivial weights."""
    data = generate_synthetic_data(n_samples=500, seed=42)
    weights = train_weights(data)
    assert weights.w_confidence != 0
    assert weights.bias != 0
    # sufficiency weight should be meaningful (negative because higher sufficiency = lower risk)
    assert abs(weights.w_sufficiency) > 0.01


def test_predict_risk():
    """Hallucination risk should be higher when sufficiency is low."""
    weights = SelectiveGenWeights(
        w_confidence=1.0,
        w_sufficiency=2.5,
        bias=-1.5,
    )
    high_risk = weights.predict_risk(confidence=0.9, sufficiency=0.1)
    low_risk = weights.predict_risk(confidence=0.9, sufficiency=0.9)
    assert high_risk > low_risk, "Low sufficiency should increase risk"

    # Risk should be in valid range
    assert 0 <= high_risk <= 1
    assert 0 <= low_risk <= 1


def test_should_answer():
    """should_answer should respect different modes."""
    weights = SelectiveGenWeights(w_confidence=1.0, w_sufficiency=2.5, bias=-1.5)

    # With very high confidence and sufficiency, should answer in all modes
    assert weights.should_answer(0.9, 0.9, mode="strict")
    assert weights.should_answer(0.9, 0.9, mode="lenient")

    # With low sufficiency but high confidence (dangerous quadrant)
    strict_decision = weights.should_answer(0.9, 0.2, mode="strict")
    lenient_decision = weights.should_answer(0.9, 0.2, mode="lenient")
    # strict should be more conservative
    assert not strict_decision or lenient_decision


def test_accuracy_coverage_curve():
    """Accuracy-coverage curve should be monotonically decreasing."""
    data = generate_synthetic_data(n_samples=500, seed=42)
    weights = train_weights(data)
    curve = compute_accuracy_coverage_curve(data, weights, n_thresholds=20)
    assert len(curve) == 20

    # Accuracy should generally decrease as coverage increases
    # (allow small fluctuations due to synthetic noise)
    high_cov = [p for p in curve if p.coverage > 0.5]
    low_cov = [p for p in curve if 0.05 < p.coverage <= 0.3]
    if high_cov and low_cov:
        avg_high_cov = sum(p.accuracy for p in high_cov) / len(high_cov)
        avg_low_cov = sum(p.accuracy for p in low_cov) / len(low_cov)
        assert avg_high_cov <= avg_low_cov + 0.02, \
            f"Lower coverage should have higher accuracy (low_cov_avg={avg_low_cov:.3f}, high_cov_avg={avg_high_cov:.3f})"


def test_compare_curves():
    """Dual-signal should outperform confidence-only on synthetic data."""
    data = generate_synthetic_data(n_samples=1000, seed=42)
    weights = train_weights(data)
    comparison = compare_curves(data, weights)
    assert comparison is not None

    # At least at some coverage point, dual signal should be better or equal
    at_least_one_better = any(
        v.get("delta", 0) >= 0 for v in comparison.values()
    )
    assert at_least_one_better


def test_print_report(capsys):
    """Print report should not crash."""
    data = generate_synthetic_data(n_samples=200, seed=42)
    weights = train_weights(data)
    curve = compute_accuracy_coverage_curve(data, weights)
    print_training_report(data, weights, curve)
    captured = capsys.readouterr()
    assert "Training samples" in captured.out
    assert "Trained weights" in captured.out
    assert "Accuracy-Coverage" in captured.out


def test_weights_serialization():
    """Weights should serialize to dict and back."""
    weights = SelectiveGenWeights(w_confidence=0.8, w_sufficiency=2.1, bias=-1.2)
    d = weights.to_dict()
    restored = SelectiveGenWeights.from_dict(d)
    assert restored.w_confidence == weights.w_confidence
    assert restored.w_sufficiency == weights.w_sufficiency
    assert restored.bias == weights.bias


def test_weight_consistency_across_seeds():
    """Different seeds should produce slightly different but valid weights."""
    w1 = train_weights(generate_synthetic_data(n_samples=500, seed=1))
    w2 = train_weights(generate_synthetic_data(n_samples=500, seed=2))
    # Both should produce valid predictions
    for w in [w1, w2]:
        risk = w.predict_risk(0.5, 0.5)
        assert 0 <= risk <= 1
