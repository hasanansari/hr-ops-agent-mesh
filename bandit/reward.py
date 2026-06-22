"""Reward computation for the bandit -- combines three independent signals
into one scalar per decision. Reuses the same 0-4 action-rank scale defined
in hitl/models.py for Section D's edit-distance math, so "how far apart are
two actions" means the same thing everywhere in the codebase.
"""

from __future__ import annotations

import numpy as np

from agents.anomaly_models import RecommendedAction
from compliance.rules_engine import evaluate as evaluate_compliance
from hitl.models import ACTION_RANK

MAX_RANK_DISTANCE = max(ACTION_RANK.values()) - min(ACTION_RANK.values())  # currently 4, derived not hardcoded

APPROVE_REWARD = 1.0
REJECT_REWARD = -1.0

# A timeout fallback is silence, not a human decision -- it must not be
# read as approval. TIMEOUT_REWARD_WEIGHT multiplies whatever the HITL
# component would otherwise have been; set to 0.0 to fully exclude timeout
# rows from training (the default and the choice used here). Left as a
# named constant rather than hardcoding the exclusion so "down-weight
# instead of exclude" is a one-line change if that's ever preferred.
TIMEOUT_REWARD_WEIGHT = 0.0

# Outcome feedback: did an auto-corrected anomaly recur? Only meaningful for
# auto-correct, the one action that claims to have actually fixed
# something rather than just routing it to a person. A genuinely real
# issue (true positive) has some baseline chance of recurring even after an
# automated fix, since auto-correct addresses the symptom (e.g. the
# payroll number) not necessarily the root cause (e.g. why it was wrong).
RECURRENCE_PROBABILITY = 0.25
RECURRENCE_PENALTY = -0.5

# False positive rate: a flagged anomaly that ground truth says was never
# real (a data error or legitimate activity, not an actual incident).
FALSE_POSITIVE_PENALTY = -0.5

# Section E is wired in for real now: does `final_action` actually trigger
# a hard compliance veto for this anomaly's context? If so, the policy
# should learn to avoid it -- -1.0 puts a compliance violation on the same
# scale as an outright human rejection, since both mean "the action picked
# was simply not acceptable," not just a matter of degree like a modify.
COMPLIANCE_VETO_PENALTY = -1.0


def compliance_veto_penalty_hook(
    anomaly_type: str, confidence: float, evidence: dict, final_action: str, employee: dict | None
) -> float:
    verdict = evaluate_compliance(
        anomaly_type=anomaly_type,
        confidence=confidence,
        evidence=evidence,
        final_action=final_action,
        employee=employee,
    )
    return COMPLIANCE_VETO_PENALTY if verdict.veto else 0.0


REWARD_CLIP = (-2.0, 1.0)


def hitl_component(human_decision: str | None, edit_distance: int | None, is_timeout_fallback: bool) -> float:
    if is_timeout_fallback:
        # whatever this would have scored, it isn't a human signal --
        # multiply down (0.0 by default = fully excluded from training).
        return APPROVE_REWARD * TIMEOUT_REWARD_WEIGHT
    if human_decision == "approve":
        return APPROVE_REWARD
    if human_decision == "reject":
        return REJECT_REWARD
    if human_decision == "modify":
        # closer rank = less penalty: a 1-step correction barely dents the
        # reward, a 4-step correction (the maximum possible) gets treated
        # almost as badly as an outright reject.
        distance = edit_distance if edit_distance is not None else MAX_RANK_DISTANCE
        return 1.0 - (distance / MAX_RANK_DISTANCE)
    raise ValueError(f"unrecognized human_decision: {human_decision!r}")


def recurrence_component(final_action: str, is_true_positive: bool, rng: np.random.Generator) -> float:
    if final_action != RecommendedAction.AUTO_CORRECT.value:
        return 0.0  # only auto-correct claims to have actually fixed anything
    if not is_true_positive:
        return 0.0  # nothing real happened, so there's nothing to recur
    recurred = rng.random() < RECURRENCE_PROBABILITY
    return RECURRENCE_PENALTY if recurred else 0.0


def false_positive_component(is_true_positive: bool) -> float:
    return 0.0 if is_true_positive else FALSE_POSITIVE_PENALTY


def combine_reward(
    *,
    human_decision: str | None,
    edit_distance: int | None,
    is_timeout_fallback: bool,
    final_action: str,
    anomaly_type: str,
    confidence: float,
    evidence: dict,
    employee: dict | None,
    is_true_positive: bool,
    rng: np.random.Generator,
) -> tuple[float, dict[str, float]]:
    """Returns (clipped_total, breakdown) -- the breakdown is kept around so
    every reward is auditable: you can always see exactly which of the four
    signals produced the final number, instead of trusting one opaque scalar.
    """
    breakdown = {
        "hitl": hitl_component(human_decision, edit_distance, is_timeout_fallback),
        "recurrence": recurrence_component(final_action, is_true_positive, rng),
        "false_positive": false_positive_component(is_true_positive),
        "compliance_veto": compliance_veto_penalty_hook(anomaly_type, confidence, evidence, final_action, employee),
    }
    total = sum(breakdown.values())
    clipped = float(np.clip(total, *REWARD_CLIP))
    return clipped, breakdown
