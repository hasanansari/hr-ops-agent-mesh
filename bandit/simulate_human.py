"""Simulates a plausible human reviewer's decision on a bandit-chosen
action. This is NOT real HITL data -- the actual decisions table (Section
D) is almost entirely timeout fallbacks so far, too sparse to train on.
This encodes one defensible heuristic instead: confidence should correlate
with trust. A human is more likely to approve a correct call when the
detector itself was confident, and more likely to reject a wrong call when
the detector was unsure -- the same intuition a real reviewer would use,
since detector confidence is the main signal they'd actually see.
"""

from __future__ import annotations

import numpy as np

from agents.anomaly_models import RecommendedAction
from hitl.models import ACTION_RANK, edit_distance

RANK_TO_ACTION = {rank: action for action, rank in ACTION_RANK.items()}


def ideal_rank(anomaly_type: str, is_true_positive: bool) -> int:
    """What a perfectly-informed reviewer (one who actually knows the
    ground truth) would consider correct. A false positive should always
    resolve to no-action -- nothing real happened. For true positives, the
    target mirrors Section B's own rule-based design: payroll always goes
    to HR (money needs sign-off), leave always goes to the manager who owns
    leave decisions, and compliance breaches are treated as auto-correctable
    in the median case -- a simplifying assumption for the simulator, since
    Section B's real rule actually splits compliance by severity.
    """
    if not is_true_positive:
        return ACTION_RANK[RecommendedAction.NO_ACTION]
    if anomaly_type == "payroll_outlier":
        return ACTION_RANK[RecommendedAction.ESCALATE_TO_HR]
    if anomaly_type == "leave_abuse":
        return ACTION_RANK[RecommendedAction.ESCALATE_TO_MANAGER]
    return ACTION_RANK[RecommendedAction.AUTO_CORRECT]  # compliance_violation


def simulate_human_decision(
    anomaly_type: str,
    confidence: float,
    chosen_action: str,
    is_true_positive: bool,
    rng: np.random.Generator,
) -> tuple[str, str, int | None]:
    """Returns (human_decision, final_action, edit_distance)."""
    chosen_rank = ACTION_RANK[RecommendedAction(chosen_action)]
    target_rank = ideal_rank(anomaly_type, is_true_positive)
    rank_gap = abs(chosen_rank - target_rank)
    correct_call = rank_gap == 0

    if correct_call:
        # approve probability rises with confidence -- a confident, correct
        # call is exactly what builds trust in the system.
        approve_prob = 0.6 + 0.35 * confidence
        if rng.random() < approve_prob:
            return "approve", chosen_action, None
        # occasionally still nudged down a notch even when technically
        # right, just to be cautious -- real reviewers aren't perfectly
        # consistent either.
        nudged = RANK_TO_ACTION[max(0, target_rank - 1)]
        return "modify", nudged, edit_distance(RecommendedAction(chosen_action), RecommendedAction(nudged))

    # wrong call -- reject probability rises as confidence FALLS, since low
    # detector confidence is itself a reason for a human to distrust the
    # call rather than bother correcting it.
    reject_prob = 0.15 + 0.55 * (1 - confidence)
    if rng.random() < reject_prob:
        return "reject", RecommendedAction.NO_ACTION.value, None

    # otherwise correct it toward what a perfectly-informed reviewer would
    # have picked.
    target_action = RANK_TO_ACTION[target_rank]
    return "modify", target_action, edit_distance(RecommendedAction(chosen_action), RecommendedAction(target_action))
