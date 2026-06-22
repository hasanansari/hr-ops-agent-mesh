"""Blends the bandit's own learned score with a bias pulled from similar
past incidents -- the actual "warm-start the RL policy from memory"
mechanism. A brand-new bandit has all-zero weights and no opinion; if
memory already holds a resolved incident that looks like the current one,
that one data point can immediately bias the choice without the bandit
needing to accumulate its own training first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bandit.policy import ACTIONS, LinearEpsilonGreedyBandit
from memory import store

# How many past incidents to retrieve for biasing.
DEFAULT_K = 5

# Memory's influence fades as the bandit accumulates its own real training,
# but never disappears completely (floored at 0.1) -- past resolved
# incidents stay informative even once the bandit has its own experience.
# 500 updates is roughly 3 full Section-B scans' worth of decisions, chosen
# as a reasonable point by which the bandit's own weights should be doing
# most of the work.
MEMORY_DECAY_UPDATES = 500
MIN_MEMORY_WEIGHT = 0.1


def memory_weight(update_count: int) -> float:
    return max(MIN_MEMORY_WEIGHT, 1.0 - update_count / MEMORY_DECAY_UPDATES)


@dataclass
class WarmStartResult:
    action: str
    margin: float  # gap between the best and second-best blended score -- a confidence proxy
    neighbor_count: int
    used_memory: bool
    bandit_scores: dict[str, float]
    memory_bias: dict[str, float]
    blended_scores: dict[str, float]


def _similarity_weighted_bias(neighbors: list[dict]) -> dict[str, float]:
    bias = {a: 0.0 for a in ACTIONS}
    weight_sum = {a: 0.0 for a in ACTIONS}
    for n in neighbors:
        # closer distance -> higher influence; +1 in the denominator keeps
        # this finite even for a near-exact match (distance ~ 0)
        similarity = 1.0 / (1.0 + n["distance"])
        action = n["action_taken"]
        bias[action] += similarity * n["reward"]
        weight_sum[action] += similarity
    for a in ACTIONS:
        if weight_sum[a] > 0:
            bias[a] /= weight_sum[a]
    return bias


def select_action_with_memory(
    bandit: LinearEpsilonGreedyBandit,
    context: np.ndarray,
    anomaly_type: str,
    confidence: float,
    evidence: dict,
    employee: dict | None,
    explore: bool = False,
    k: int = DEFAULT_K,
    collection_name: str = store.DEFAULT_COLLECTION,
) -> WarmStartResult:
    if explore and bandit.rng.random() < bandit.epsilon:
        action = str(bandit.rng.choice(ACTIONS))
        bandit_scores = {a: bandit.predict(context, a) for a in ACTIONS}
        return WarmStartResult(action, margin=0.0, neighbor_count=0, used_memory=False,
                                bandit_scores=bandit_scores, memory_bias={a: 0.0 for a in ACTIONS},
                                blended_scores=bandit_scores)

    neighbors = store.query_similar(anomaly_type, confidence, evidence, employee, k=k, collection_name=collection_name)
    bandit_scores = {a: bandit.predict(context, a) for a in ACTIONS}

    if not neighbors:
        blended = dict(bandit_scores)
        memory_bias = {a: 0.0 for a in ACTIONS}
    else:
        memory_bias = _similarity_weighted_bias(neighbors)
        weight = memory_weight(bandit.update_count)
        blended = {a: bandit_scores[a] + weight * memory_bias[a] for a in ACTIONS}

    ranked = sorted(blended.values(), reverse=True)
    best_score = ranked[0]
    second_score = ranked[1] if len(ranked) > 1 else ranked[0]
    margin = best_score - second_score

    tied = [a for a, s in blended.items() if s == best_score]
    action = str(bandit.rng.choice(tied))

    return WarmStartResult(
        action=action,
        margin=margin,
        neighbor_count=len(neighbors),
        used_memory=len(neighbors) > 0,
        bandit_scores=bandit_scores,
        memory_bias=memory_bias,
        blended_scores=blended,
    )
