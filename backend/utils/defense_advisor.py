"""Defense Advisor — recommends defenses based on attack results."""

from __future__ import annotations


def recommend_defenses(attack_results: dict[str, dict]) -> list[dict]:
    """Analyse attack results and recommend defenses with priorities.

    Rules
    -----
    - White-box damage > 40 % → Adversarial Training (Strong)
    - Black-box damage > 30 % → Feature Squeezing + Quick Adversarial Training
    - C&W adv_accuracy < 20 % → Randomized Smoothing (Certified)
    - Transfer attack succeeds → Ensemble + Preprocessing

    Parameters
    ----------
    attack_results : dict[str, dict]
        Mapping of attack name → result dict with ``clean_accuracy``
        and ``adv_accuracy`` keys.

    Returns
    -------
    list[dict]
        Sorted list of recommendations, each with keys:
        ``{"defense", "priority", "reason", "config"}``.
        Priority: ``"critical"`` > ``"high"`` > ``"medium"`` > ``"low"``.
    """
    recommendations: list[dict] = []

    white_box_attacks = {"FGSM", "PGD", "C&W"}
    black_box_attacks = {"HopSkipJump", "Square"}

    for name, result in attack_results.items():
        clean = result.get("clean_accuracy", 1.0)
        adv = result.get("adv_accuracy", clean)
        if clean <= 0:
            continue
        damage = (clean - adv) / clean

        # White-box damage > 40%
        if name in white_box_attacks and damage > 0.4:
            recommendations.append({
                "defense": "Adversarial Training",
                "priority": "critical",
                "reason": (
                    f"{name} reduced accuracy by {damage:.0%}. "
                    "Adversarial training with PGD is the strongest known defense "
                    "against gradient-based attacks."
                ),
                "config": {
                    "type": "adversarial_training",
                    "epsilon": result.get("epsilon", 0.3),
                    "nb_epochs": 10,
                    "ratio": 0.5,
                },
            })

        # C&W adv_accuracy < 20%
        if name == "C&W" and adv < 0.2:
            recommendations.append({
                "defense": "Randomized Smoothing",
                "priority": "critical",
                "reason": (
                    f"C&W drove accuracy to {adv:.0%}. "
                    "Randomized smoothing provides certified robustness "
                    "against L2-bounded adversaries."
                ),
                "config": {
                    "type": "randomized_smoothing",
                    "sigma": 0.25,
                    "sample_size": 100,
                },
            })

        # Black-box damage > 30%
        if name in black_box_attacks and damage > 0.3:
            recommendations.append({
                "defense": "Feature Squeezing",
                "priority": "high",
                "reason": (
                    f"{name} reduced accuracy by {damage:.0%}. "
                    "Feature squeezing removes fine-grained perturbations "
                    "that black-box attacks rely on."
                ),
                "config": {
                    "type": "feature_squeezing",
                    "bit_depth": 4,
                },
            })
            recommendations.append({
                "defense": "Quick Adversarial Training",
                "priority": "high",
                "reason": (
                    f"Fast adversarial fine-tuning to harden against "
                    f"{name}-style black-box perturbations."
                ),
                "config": {
                    "type": "adversarial_training",
                    "epsilon": result.get("epsilon", 0.3),
                    "nb_epochs": 3,
                    "ratio": 0.3,
                },
            })

        # Transfer attack success
        if name == "Transfer" and damage > 0.1:
            recommendations.append({
                "defense": "Input Preprocessing",
                "priority": "medium",
                "reason": (
                    f"Transfer attack succeeded with {damage:.0%} damage. "
                    "JPEG compression and Gaussian smoothing remove "
                    "transferable perturbation patterns."
                ),
                "config": {
                    "type": "jpeg",
                    "quality": 50,
                },
            })
            recommendations.append({
                "defense": "Ensemble Diversity",
                "priority": "medium",
                "reason": (
                    "Transfer attacks exploit shared decision boundaries. "
                    "Using diverse model architectures reduces transferability."
                ),
                "config": {
                    "type": "ensemble",
                    "note": "Train 3+ models with different architectures.",
                },
            })

    # Deduplicate by defense name, keeping highest priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    seen: dict[str, int] = {}
    deduped: list[dict] = []
    for rec in recommendations:
        key = rec["defense"]
        pri = priority_order.get(rec["priority"], 3)
        if key not in seen or pri < seen[key]:
            if key in seen:
                deduped = [r for r in deduped if r["defense"] != key]
            seen[key] = pri
            deduped.append(rec)

    deduped.sort(key=lambda r: priority_order.get(r["priority"], 3))

    # Always include a baseline recommendation
    if not deduped:
        deduped.append({
            "defense": "Model Monitoring",
            "priority": "low",
            "reason": (
                "No critical vulnerabilities detected. "
                "Continue monitoring with periodic robustness evaluations."
            ),
            "config": {"type": "monitoring"},
        })

    return deduped
