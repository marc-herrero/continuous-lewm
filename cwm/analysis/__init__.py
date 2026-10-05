"""
cwm.analysis: Evaluation and diagnostic statistics.
"""

from .statistics import mcnemar_test, wilson_score_interval

__all__ = ["mcnemar_test", "wilson_score_interval"]
