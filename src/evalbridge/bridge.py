"""Week 2 (not built yet): the bridge study.

Question: when a judge says model X beats model Y by some margin, how much more do real users
prefer X? Plan:

1. Score Chatbot Arena model pairs with the offline judge (judge win rate per pair).
2. Take human preference for the same pairs from Arena battles
   (``lmsys/chatbot_arena_conversations``).
3. Regress the change in human preference on the change in judge score across pairs, with
   bootstrap CIs, and report how much of the variance the judge explains.
"""

from __future__ import annotations

import pandas as pd


def pair_deltas(judge_scores: pd.DataFrame, human_scores: pd.DataFrame) -> pd.DataFrame:
    """Join per-pair judge and human win rates into one row per model pair."""
    raise NotImplementedError("Week 2")


def fit_bridge(deltas: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Slope, intercept and R^2 of human delta on judge delta, with bootstrap CIs."""
    raise NotImplementedError("Week 2")
