"""
Statistical evaluation utilities:
- Exact two-sided McNemar tests on paired binary outcomes.
- 95% Wilson Score confidence intervals for binomial proportions.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes the Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)
    return lower * 100.0, upper * 100.0


def mcnemar_test(vec_a: List[bool], vec_b: List[bool]) -> Dict[str, float]:
    """
    Computes exact McNemar test on paired binary outcome vectors.
    vec_a: Baseline outcomes (e.g. discrete model)
    vec_b: Test outcomes (e.g. continuous model)
    """
    assert len(vec_a) == len(vec_b), f"Length mismatch: {len(vec_a)} vs {len(vec_b)}"
    a = np.array(vec_a, dtype=bool)
    b = np.array(vec_b, dtype=bool)

    n11 = int(np.sum(a & b))
    n00 = int(np.sum((~a) & (~b)))
    b_disc = int(np.sum(a & (~b)))  # Baseline wins, test loses
    c_disc = int(np.sum((~a) & b))  # Test wins, baseline loses
    disc_total = b_disc + c_disc

    if disc_total == 0:
        p_val = 1.0
        chi2_stat = 0.0
    else:
        p_val = stats.binomtest(min(b_disc, c_disc), disc_total, 0.5, alternative="two-sided").pvalue
        chi2_stat = ((abs(b_disc - c_disc) - 1.0) ** 2) / disc_total

    return {
        "n11_both_success": n11,
        "n00_both_fail": n00,
        "b_baseline_wins": b_disc,
        "c_test_wins": c_disc,
        "discordant_total": disc_total,
        "net_gain": c_disc - b_disc,
        "chi2": chi2_stat,
        "p_value": p_val,
    }
