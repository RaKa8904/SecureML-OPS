"""Unit tests for the robustness scorer."""

from __future__ import annotations

import pytest

from backend.utils.scorer import ATTACK_WEIGHTS, compute_robustness_score


class TestRobustnessScore:
    def _make_result(self, adv_accuracy: float) -> dict:
        return {"adv_accuracy": adv_accuracy, "clean_accuracy": 0.95}

    def test_perfect_robustness(self) -> None:
        results = {a: self._make_result(0.95) for a in ATTACK_WEIGHTS}
        out = compute_robustness_score(results, clean_accuracy=0.95)
        assert out["score"] == 100.0
        assert out["severity"] == "EXCELLENT"

    def test_zero_robustness(self) -> None:
        results = {a: self._make_result(0.0) for a in ATTACK_WEIGHTS}
        out = compute_robustness_score(results, clean_accuracy=0.95)
        assert out["score"] == 0.0
        assert out["severity"] == "CRITICAL"

    def test_partial_attacks(self) -> None:
        results = {
            "FGSM": self._make_result(0.5),
            "PGD": self._make_result(0.3),
        }
        out = compute_robustness_score(results, clean_accuracy=0.95)
        assert 0 < out["score"] < 100
        assert "breakdown" in out

    def test_severity_levels(self) -> None:
        for adv, expected in [(0.25, "CRITICAL"), (0.40, "HIGH"),
                               (0.60, "MODERATE"), (0.75, "STRONG"),
                               (0.94, "EXCELLENT")]:
            results = {a: self._make_result(adv) for a in ATTACK_WEIGHTS}
            out = compute_robustness_score(results, clean_accuracy=0.95)
            assert out["severity"] == expected, f"adv={adv} → {out['severity']} != {expected}"

    def test_zero_clean_accuracy(self) -> None:
        results = {"FGSM": self._make_result(0.0)}
        out = compute_robustness_score(results, clean_accuracy=0.0)
        assert out["score"] == 0.0
        assert out["severity"] == "CRITICAL"

    def test_breakdown_passed_through(self) -> None:
        results = {"FGSM": self._make_result(0.5)}
        out = compute_robustness_score(results, clean_accuracy=0.95)
        assert out["breakdown"] is results
