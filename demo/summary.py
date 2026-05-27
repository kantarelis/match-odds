"""Turn calibrated probabilities into a one-line, plain-English read for the demo.

Logic lives here (not in ``app.py``) so it is unit-testable without a running browser, mirroring the
repo's "UI stays thin; logic is importable" rule.
"""

from __future__ import annotations

from demo.service import Probabilities

# Below this lead (in probability points) over the runner-up we call it a coin-flip rather than
# name a favourite — honest about the model's uncertainty on a tight fixture.
_CLOSE_LEAD = 0.10

_OUTCOME_PHRASES: dict[str, str] = {
    "home_win": "a home win",
    "draw": "a draw",
    "away_win": "an away win",
}


def verbal_summary(probs: Probabilities) -> str:
    """Name the most likely outcome with its percentage, or call a tight spread a close match."""
    outcomes = {"home_win": probs.home_win, "draw": probs.draw, "away_win": probs.away_win}
    leader, top = max(outcomes.items(), key=lambda item: item[1])
    runner_up = sorted(outcomes.values(), reverse=True)[1]
    if top - runner_up < _CLOSE_LEAD:
        return f"Too close to call — no clear favourite (top outcome only {top * 100:.0f}%)."
    return f"Model favours {_OUTCOME_PHRASES[leader]} at {top * 100:.0f}%."
