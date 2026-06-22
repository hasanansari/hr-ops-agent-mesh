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
from hitl.models import ACTION_RANK, RANK_TO_ACTION, edit_distance


def ideal_rank(anomaly_type: str, is_true_positive: bool, evidence: dict | None = None) -> int:
    """What a perfectly-informed reviewer (one who actually knows the
    ground truth) would consider correct. A false positive should always
    resolve to no-action -- nothing real happened. For true positives,
    payroll always goes to HR (money needs sign-off) and leave always goes
    to the manager who owns leave decisions, mirroring Section B's own
    rule-based design.

    Compliance violations are split by sub-type to match Section E's actual
    hard rules, not flattened to one constant -- an earlier version of this
    function always called compliance "ideal" at auto-correct, which
    directly contradicted Section E's TRAINING_CANNOT_BE_AUTO_CORRECTED rule
    and made the bandit look like it was getting *worse* after learning,
    when really two of this project's own modules just disagreed with each
    other about what "correct" meant for that one sub-type.
    """
    if not is_true_positive:
        return ACTION_RANK[RecommendedAction.NO_ACTION]
    if anomaly_type == "payroll_outlier":
        return ACTION_RANK[RecommendedAction.ESCALATE_TO_HR]
    if anomaly_type == "leave_abuse":
        return ACTION_RANK[RecommendedAction.ESCALATE_TO_MANAGER]

    # compliance_violation -- mirrors Section B's own severity-based split
    # for overtime, and Section E's hard rule that training can never be
    # auto-corrected.
    evidence = evidence or {}
    if evidence.get("violation") == "missing_mandatory_training":
        return ACTION_RANK[RecommendedAction.ESCALATE_TO_MANAGER]
    severity = evidence.get("severity", 1.0)
    if severity < 0.5:
        return ACTION_RANK[RecommendedAction.AUTO_CORRECT]
    return ACTION_RANK[RecommendedAction.ESCALATE_TO_HR]


def simulate_human_decision(
    anomaly_type: str,
    confidence: float,
    chosen_action: str,
    is_true_positive: bool,
    rng: np.random.Generator,
    evidence: dict | None = None,
) -> tuple[str, str, int | None]:
    """Returns (human_decision, final_action, edit_distance)."""
    chosen_rank = ACTION_RANK[RecommendedAction(chosen_action)]
    target_rank = ideal_rank(anomaly_type, is_true_positive, evidence)
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
