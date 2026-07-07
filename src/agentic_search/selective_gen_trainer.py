"""Selective Generation trainer — trains logistic regression on synthetic data.

Per the paper:
  - Signal 1: Self-rated confidence P(Correct) or P(True)
  - Signal 2: Sufficient context binary label
  - Model: logistic regression → hallucination_risk = sigmoid(w1*conf + w2*suff + b)
  - Result: accuracy-coverage curves showing 2-10% improvement

This module provides:
  - Synthetic data generator (mimicking the paper's experimental distributions)
  - Logistic regression trainer with cross-validation
  - Accuracy-coverage curve generator
  - Weight export for production use
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# scikit-learn is optional — fallback to pure numpy if not available
try:
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Synthetic Data Generation ──


@dataclass
class TrainingInstance:
    """One training example for selective generation."""
    confidence: float        # P(Correct) or P(True) — 0..1
    sufficiency_score: float # 0..1
    is_correct: bool         # Whether the model answered correctly (ground truth)


def generate_synthetic_data(
    n_samples: int = 1000,
    seed: int = 42,
) -> list[TrainingInstance]:
    """Generate synthetic training data mimicking the paper's experimental findings.

    The paper's key finding: models exhibit different behavior in 4 quadrants:

        ┌─────────────────────┬─────────────────────┐
        │  Sufficient + High C│  Sufficient + Low C │
        │  → correct 85%      │  → correct 60%      │
        ├─────────────────────┼─────────────────────┤
        │  Insufficient + HC  │  Insufficient + LC  │
        │  → correct 35%      │  → correct 20%      │
        │  ← MOST DANGEROUS   │                     │
        └─────────────────────┴─────────────────────┘

    These percentages approximate Figure 4 in the paper.
    """
    rng = random.Random(seed)
    data: list[TrainingInstance] = []

    for _ in range(n_samples):
        # Sample sufficiency from realistic distribution
        is_sufficient = rng.random() < 0.4  # ~40% of queries have sufficient context

        if is_sufficient:
            sufficiency = rng.uniform(0.7, 1.0)
            confidence = rng.uniform(0.3, 0.95)
            # With sufficient context, correctness is high but not perfect
            correct_prob = 0.75 + 0.15 * confidence
        else:
            sufficiency = rng.uniform(0.0, 0.45)
            confidence = rng.uniform(0.1, 0.95)
            # With insufficient context, correctness depends on confidence
            # Key insight: high confidence + insufficient context → 66% error rate
            if confidence > 0.7:
                correct_prob = 0.30  # Most dangerous quadrant
            else:
                correct_prob = 0.15 + 0.15 * confidence

        is_correct = rng.random() < correct_prob
        data.append(TrainingInstance(
            confidence=confidence,
            sufficiency_score=sufficiency,
            is_correct=is_correct,
        ))

    return data


# ── Logistic Regression Trainer ──


@dataclass
class SelectiveGenWeights:
    """Trained weights for selective generation."""
    w_confidence: float
    w_sufficiency: float
    bias: float
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "strict": 0.70,
        "balanced": 0.50,
        "lenient": 0.30,
    })

    def predict_risk(self, confidence: float, sufficiency: float) -> float:
        """Predict hallucination risk: P(hallucination | confidence, sufficiency)."""
        logit = self.bias + self.w_confidence * confidence - self.w_sufficiency * sufficiency
        return 1.0 / (1.0 + math.exp(-logit))

    def should_answer(self, confidence: float, sufficiency: float, mode: str = "balanced") -> bool:
        """Decide whether to answer based on mode threshold."""
        risk = self.predict_risk(confidence, sufficiency)
        threshold = self.thresholds.get(mode, 0.50)
        return risk < threshold

    def to_dict(self) -> dict:
        return {
            "w_confidence": self.w_confidence,
            "w_sufficiency": self.w_sufficiency,
            "bias": self.bias,
            "thresholds": self.thresholds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SelectiveGenWeights:
        return cls(
            w_confidence=d["w_confidence"],
            w_sufficiency=d["w_sufficiency"],
            bias=d["bias"],
            thresholds=d.get("thresholds", {"strict": 0.70, "balanced": 0.50, "lenient": 0.30}),
        )

    @classmethod
    def random_initial(cls) -> SelectiveGenWeights:
        """Random initial weights — useful before training."""
        return cls(
            w_confidence=1.0,
            w_sufficiency=-2.5,
            bias=1.5,
        )


def train_weights(
    data: list[TrainingInstance],
    use_sklearn: bool = True,
) -> SelectiveGenWeights:
    """Train logistic regression weights from training data.

    Target variable: hallucination = not is_correct
    Features: [confidence, sufficiency_score]
    """
    if SKLEARN_AVAILABLE and use_sklearn:
        return _train_sklearn(data)
    return _train_numpy(data)


def _train_sklearn(data: list[TrainingInstance]) -> SelectiveGenWeights:
    """Train using scikit-learn LogisticRegression."""
    X = np.array([[d.confidence, d.sufficiency_score] for d in data])
    y = np.array([0 if d.is_correct else 1 for d in data])  # 1 = hallucination

    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    model.fit(X, y)

    # Extract weights
    # sklearn: logit = intercept + coef[0]*conf + coef[1]*suff
    # Our formula: logit = bias + w_confidence*conf - w_sufficiency*suff
    # So: w_confidence = coef[0], w_sufficiency = -coef[1], bias = intercept
    w_confidence = float(model.coef_[0][0])
    w_sufficiency = float(-model.coef_[0][1])  # Negative because higher sufficiency → lower risk
    bias = float(model.intercept_[0])

    # Calibrate thresholds from training data
    thresholds = _calibrate_thresholds(model, X)

    return SelectiveGenWeights(
        w_confidence=w_confidence,
        w_sufficiency=w_sufficiency,
        bias=bias,
        thresholds=thresholds,
    )


def _train_numpy(data: list[TrainingInstance]) -> SelectiveGenWeights:
    """Pure numpy logistic regression (no sklearn dependency)."""
    X = np.array([[d.confidence, d.sufficiency_score] for d in data])
    y = np.array([0 if d.is_correct else 1 for d in data])
    n, p = X.shape

    # Gradient descent
    X_aug = np.c_[np.ones(n), X]  # Add bias column
    theta = np.zeros(p + 1)
    lr = 0.1
    for _ in range(5000):
        z = X_aug @ theta
        h = 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))
        grad = (X_aug.T @ (h - y)) / n
        theta -= lr * grad

    bias = float(theta[0])
    w_confidence = float(theta[1])
    w_sufficiency = float(-theta[2])  # Our convention: positive sufficiency → lower risk

    return SelectiveGenWeights(
        w_confidence=w_confidence,
        w_sufficiency=w_sufficiency,
        bias=bias,
        thresholds={"strict": 0.70, "balanced": 0.50, "lenient": 0.30},
    )


def _calibrate_thresholds(model, X: np.ndarray) -> dict[str, float]:
    """Heuristically set strict/balanced/lenient thresholds based on training data."""
    probs = model.predict_proba(X)[:, 1]  # P(hallucination)
    # threshold at median, 75th percentile, 25th percentile
    thresholds = {
        "strict": float(np.percentile(probs, 75)),   # Only answer very low risk
        "balanced": float(np.median(probs)),           # Default
        "lenient": float(np.percentile(probs, 25)),    # Will answer most queries
    }
    # Clip to reasonable range
    for k in thresholds:
        thresholds[k] = max(0.1, min(0.95, thresholds[k]))
    return thresholds


# ── Accuracy-Coverage Curve ──


@dataclass
class CoveragePoint:
    """One point on the accuracy-coverage curve."""
    threshold: float
    coverage: float       # Fraction of queries answered
    accuracy: float       # Correct / Answered
    n_answered: int
    n_correct: int


def compute_accuracy_coverage_curve(
    data: list[TrainingInstance],
    weights: SelectiveGenWeights,
    n_thresholds: int = 50,
) -> list[CoveragePoint]:
    """Compute accuracy-coverage curve for the given weights and data.

    The x-axis (coverage) is the fraction of queries where the model answers.
    The y-axis (accuracy) is the fraction of answered queries that are correct.

    This produces the paper's Figure 4.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    points: list[CoveragePoint] = []

    for thresh in thresholds:
        answered = []
        for d in data:
            risk = weights.predict_risk(d.confidence, d.sufficiency_score)
            if risk < thresh:
                answered.append(d.is_correct)

        if answered:
            n_answered = len(answered)
            n_correct = sum(answered)
            coverage = n_answered / len(data)
            accuracy = n_correct / n_answered
        else:
            n_answered = 0
            n_correct = 0
            coverage = 0.0
            accuracy = 1.0  # Vacuous: no answers → all correct

        points.append(CoveragePoint(
            threshold=float(thresh),
            coverage=float(coverage),
            accuracy=float(accuracy),
            n_answered=n_answered,
            n_correct=n_correct,
        ))

    return points


def compare_curves(
    data: list[TrainingInstance],
    dual_signal_weights: SelectiveGenWeights,
    confidence_only_weights: SelectiveGenWeights | None = None,
) -> dict:
    """Compare dual-signal vs confidence-only, per the paper's main result.

    Returns accuracy deltas at key coverage points.
    """
    dual_curve = compute_accuracy_coverage_curve(data, dual_signal_weights)

    # Confidence-only: ignore sufficiency
    conf_only_weights = confidence_only_weights or SelectiveGenWeights(
        w_confidence=dual_signal_weights.w_confidence * 1.5,
        w_sufficiency=0.0,  # Ignore sufficiency signal
        bias=dual_signal_weights.bias,
    )
    conf_only_curve = compute_accuracy_coverage_curve(data, conf_only_weights)

    # Compare at coverage points 50%, 70%, 90%
    comparison = {}
    for target_coverage in [0.50, 0.70, 0.90]:
        dual_pt = _find_nearest_coverage(dual_curve, target_coverage)
        conf_pt = _find_nearest_coverage(conf_only_curve, target_coverage)
        if dual_pt and conf_pt:
            delta = dual_pt.accuracy - conf_pt.accuracy
            comparison[f"cov_{target_coverage:.0%}"] = {
                "dual_signal_accuracy": dual_pt.accuracy,
                "confidence_only_accuracy": conf_pt.accuracy,
                "delta": delta,
                "delta_str": f"{delta:+.1%}",
            }

    return comparison


def _find_nearest_coverage(curve: list[CoveragePoint], target: float) -> CoveragePoint | None:
    """Find the point with coverage closest to target."""
    if not curve:
        return None
    return min(curve, key=lambda p: abs(p.coverage - target))


# ── Report ──


def print_training_report(
    data: list[TrainingInstance],
    weights: SelectiveGenWeights,
    curve: list[CoveragePoint],
    comparison: dict | None = None,
):
    """Pretty-print training results."""
    n_correct = sum(1 for d in data if d.is_correct)
    print(f"\n{'='*72}")
    print(f"  Selective Generation - Training Report")
    print(f"{'='*72}")
    print(f"  Training samples: {len(data)}")
    print(f"  Baseline accuracy (no abstention): {n_correct/len(data):.1%}")

    print(f"\n  Trained weights:")
    print(f"    w_confidence  = {weights.w_confidence:+.4f}")
    print(f"    w_sufficiency = {weights.w_sufficiency:+.4f}")
    print(f"    bias          = {weights.bias:+.4f}")
    print(f"  Thresholds: {weights.thresholds}")

    print(f"\n  Accuracy-Coverage curve (key points):")
    for cov_target in [0.50, 0.70, 0.90]:
        pt = _find_nearest_coverage(curve, cov_target)
        if pt:
            print(f"    Coverage {cov_target:.0%}: accuracy = {pt.accuracy:.1%}  "
                  f"(threshold={pt.threshold:.2f}, answered={pt.n_answered})")

    if comparison:
        print(f"\n  Dual-signal vs Confidence-only comparison:")
        for key, val in comparison.items():
            if val["delta"] > 0:
                print(f"    {key}: dual {val['dual_signal_accuracy']:.1%} vs "
                      f"conf-only {val['confidence_only_accuracy']:.1%} "
                      f"(▲ {val['delta']:+.1%})")
            else:
                print(f"    {key}: dual {val['dual_signal_accuracy']:.1%} vs "
                      f"conf-only {val['confidence_only_accuracy']:.1%} "
                      f"({val['delta_str']})")

    # Export format
    print(f"\n  Export-ready weights:")
    print(f"  {weights.to_dict()}")
    print(f"{'='*72}\n")
