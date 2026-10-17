"""Evidence ladder (hyp.txt section 12).

Five levels — never a fake "AI confidence %":
Strong / Moderate / Limited / Insufficient / Contradictory.
Wraps investigate._strength_for thresholds; adds the red level when
contradictory signals outweigh support.
"""

from __future__ import annotations

ORDER = ["Strong evidence", "Moderate evidence", "Limited evidence",
         "Insufficient evidence", "Contradictory evidence"]

_RANK = {k: i for i, k in enumerate(ORDER)}


def normalize(strength: str, n_support: int = 0, n_against: int = 0) -> str:
    """Map any legacy label to the 5-level ladder.

    Contradictory wins only when evidence-against strictly outnumbers
    support (both measured, never LLM-voted).
    """
    s = (strength or "").strip()
    if s not in _RANK:
        s = "Insufficient evidence"
    if n_against > n_support:
        return "Contradictory evidence"
    return s


def rank_key(strength: str) -> int:
    return _RANK.get(strength, 3)
