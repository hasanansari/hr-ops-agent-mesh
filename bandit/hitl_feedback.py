"""Ingests REAL persisted HITL decisions (Section D's SQLite store) as
training data -- the actual wiring point between Section D and Section C.
Today the store is almost entirely timeout fallbacks (no real review has
happened yet against this dataset), so in practice this contributes little
training signal -- but that's exactly what should happen: timeout rows are
excluded from the reward (TIMEOUT_REWARD_WEIGHT = 0.0 in reward.py), so
this function correctly learns nothing from silence. Any genuine
approve/reject/modify made through hitl/app.py flows into here exactly the
same way.
"""

from __future__ import annotations

import json

import numpy as np

from agents.anomaly_models import RecommendedAction
from bandit.ground_truth import ground_truth_lookup, is_true_positive
from bandit.policy import LinearEpsilonGreedyBandit, context_vector
from bandit.reward import combine_reward
from data.generate_employees import generate_employees
from hitl import store


def load_real_decisions() -> list[dict]:
    return [row for row in store.list_all() if row["status"] != "pending"]


def train_on_real_decisions(bandit: LinearEpsilonGreedyBandit, rng: np.random.Generator) -> list[dict]:
    decisions = load_real_decisions()
    if not decisions:
        return []

    _, truth = generate_employees()
    ground_truth_by_id = ground_truth_lookup(truth)

    log = []
    for row in decisions:
        if row["employee_id"] not in ground_truth_by_id:
            continue  # references a record outside the canonical seeded dataset, can't grade it
        evidence = json.loads(row["evidence_json"])
        is_tp = is_true_positive(row["anomaly_type"], evidence, row["employee_id"], ground_truth_by_id)
        context = context_vector(row["anomaly_type"], row["confidence"])

        reward, breakdown = combine_reward(
            human_decision=row["human_decision"],
            edit_distance=row["edit_distance"],
            is_timeout_fallback=bool(row["is_timeout_fallback"]),
            final_action=row["final_action"] or RecommendedAction.NO_ACTION.value,
            anomaly_type=row["anomaly_type"],
            is_true_positive=is_tp,
            rng=rng,
        )
        # credit/blame the action the policy actually proposed at the time,
        # not the human's correction -- that's what the weights should learn from.
        bandit.update(context, row["proposed_action"], reward)

        log.append(
            {
                "anomaly_id": row["anomaly_id"],
                "anomaly_type": row["anomaly_type"],
                "proposed_action": row["proposed_action"],
                "human_decision": row["human_decision"],
                "is_timeout_fallback": bool(row["is_timeout_fallback"]),
                "reward": reward,
                "reward_breakdown": breakdown,
            }
        )
    return log
