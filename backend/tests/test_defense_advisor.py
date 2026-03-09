"""Tests for defense_advisor utility."""

from __future__ import annotations

import pytest

from backend.utils.defense_advisor import recommend_defenses


class TestRecommendDefenses:
    def test_whitebox_high_damage(self):
        results = {
            "PGD": {"clean_accuracy": 0.95, "adv_accuracy": 0.30, "epsilon": 0.3},
        }
        recs = recommend_defenses(results)
        assert any(r["defense"] == "Adversarial Training" for r in recs)
        assert recs[0]["priority"] == "critical"

    def test_cw_low_accuracy(self):
        results = {
            "C&W": {"clean_accuracy": 0.95, "adv_accuracy": 0.10, "epsilon": 0.3},
        }
        recs = recommend_defenses(results)
        defenses = [r["defense"] for r in recs]
        assert "Randomized Smoothing" in defenses

    def test_blackbox_high_damage(self):
        results = {
            "Square": {"clean_accuracy": 0.90, "adv_accuracy": 0.40, "epsilon": 0.3},
        }
        recs = recommend_defenses(results)
        defenses = [r["defense"] for r in recs]
        assert "Feature Squeezing" in defenses

    def test_transfer_success(self):
        results = {
            "Transfer": {"clean_accuracy": 0.90, "adv_accuracy": 0.60, "epsilon": 0.3},
        }
        recs = recommend_defenses(results)
        defenses = [r["defense"] for r in recs]
        assert "Input Preprocessing" in defenses
        assert "Ensemble Diversity" in defenses

    def test_no_vulnerabilities(self):
        results = {
            "FGSM": {"clean_accuracy": 0.95, "adv_accuracy": 0.90, "epsilon": 0.3},
        }
        recs = recommend_defenses(results)
        assert len(recs) == 1
        assert recs[0]["defense"] == "Model Monitoring"
        assert recs[0]["priority"] == "low"

    def test_multiple_attacks_combined(self):
        results = {
            "PGD": {"clean_accuracy": 0.95, "adv_accuracy": 0.20, "epsilon": 0.3},
            "C&W": {"clean_accuracy": 0.95, "adv_accuracy": 0.15, "epsilon": 0.3},
            "Square": {"clean_accuracy": 0.90, "adv_accuracy": 0.30, "epsilon": 0.3},
            "Transfer": {"clean_accuracy": 0.90, "adv_accuracy": 0.50, "epsilon": 0.3},
        }
        recs = recommend_defenses(results)
        assert len(recs) >= 3
        assert recs[0]["priority"] == "critical"

    def test_dedup_keeps_highest_priority(self):
        results = {
            "PGD": {"clean_accuracy": 0.95, "adv_accuracy": 0.20, "epsilon": 0.3},
            "Square": {"clean_accuracy": 0.90, "adv_accuracy": 0.30, "epsilon": 0.3},
        }
        recs = recommend_defenses(results)
        adv_training_recs = [r for r in recs if r["defense"] == "Adversarial Training"]
        assert len(adv_training_recs) <= 1

    def test_zero_clean_accuracy_skipped(self):
        results = {
            "PGD": {"clean_accuracy": 0.0, "adv_accuracy": 0.0, "epsilon": 0.3},
        }
        recs = recommend_defenses(results)
        assert recs[0]["defense"] == "Model Monitoring"

    def test_config_keys(self):
        results = {
            "PGD": {"clean_accuracy": 0.95, "adv_accuracy": 0.20, "epsilon": 0.3},
        }
        recs = recommend_defenses(results)
        for rec in recs:
            assert "defense" in rec
            assert "priority" in rec
            assert "reason" in rec
            assert "config" in rec
