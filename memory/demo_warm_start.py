"""The required Section F deliverable: prove the same anomaly type is
handled with higher confidence on its second occurrence than its first,
using episodic memory alone -- not bandit training. Deliberately distinct
from Section C's train_cycles.py demo: that script proves the bandit's OWN
weights improve after many gradient updates. This script proves something
different and specific to memory: a SINGLE resolved incident can
immediately bias the very next similar one, with the bandit's weights
completely untouched (zero .update() calls) between the two occurrences.

Uses an isolated Chroma collection so this demo is fully reproducible and
never entangled with the shared "episodic_memory" collection that
train_cycles.py and the live graph actually use.
"""

from __future__ import annotations

import numpy as np

from agents.anomaly_scoring import run_anomaly_scan
from bandit.ground_truth import ground_truth_lookup, is_true_positive
from bandit.policy import LinearEpsilonGreedyBandit, context_vector
from bandit.reward import combine_reward
from bandit.simulate_human import simulate_human_decision
from data.generate_employees import generate_employees
from memory import store
from memory.warm_start import select_action_with_memory

DEMO_COLLECTION = "warm_start_demo"


def _pick_similar_pair(anomalies: list[dict], ground_truth_by_id: dict[str, dict]) -> tuple[dict, dict]:
    """Finds the two real payroll_outlier anomalies from the actual Section
    B scan whose z-scores are closest to each other -- a genuinely similar
    pair pulled from real data, not a constructed example. Restricted to
    true positives (real injected outliers, per Section B's ground truth):
    a false positive has no "ideal" action at all, so a single resolved
    false-positive incident can only teach memory to avoid one bad choice,
    not to confidently prefer a good one -- the wrong example for proving
    a *higher-confidence* second occurrence specifically.
    """
    payroll = [
        a for a in anomalies
        if a["anomaly_type"] == "payroll_outlier" and ground_truth_by_id[a["employee_id"]]["is_payroll_outlier"]
    ]
    payroll.sort(key=lambda a: a["evidence"]["z_score"])
    best_pair, best_gap = None, float("inf")
    for a, b in zip(payroll, payroll[1:]):
        gap = abs(a["evidence"]["z_score"] - b["evidence"]["z_score"])
        if gap < best_gap:
            best_gap, best_pair = gap, (a, b)
    return best_pair


def main() -> None:
    # fully reset the demo collection so this script's result is
    # reproducible no matter what's accumulated from previous runs
    store.reset_collection(DEMO_COLLECTION)

    employees, truth = generate_employees()
    ground_truth_by_id = ground_truth_lookup(truth)
    employees_by_id = {e["employee_id"]: e for e in employees}

    scan = run_anomaly_scan(employees)
    anomalies = scan["high_confidence_anomalies"] + scan["review_queue"]
    occurrence_1, occurrence_2 = _pick_similar_pair(anomalies, ground_truth_by_id)
    print(
        f"chosen pair (closest z-scores among real payroll_outlier anomalies): "
        f"{occurrence_1['employee_id']} (z={occurrence_1['evidence']['z_score']:+.2f}) and "
        f"{occurrence_2['employee_id']} (z={occurrence_2['evidence']['z_score']:+.2f})"
    )

    bandit = LinearEpsilonGreedyBandit(epsilon=0.0, learning_rate=0.1, seed=1)
    rng = np.random.default_rng(7)

    def resolve_and_record(anomaly: dict, result) -> dict:
        is_tp = is_true_positive(anomaly["anomaly_type"], anomaly["evidence"], anomaly["employee_id"], ground_truth_by_id)
        employee = employees_by_id.get(anomaly["employee_id"])
        human_decision, final_action, edit_dist = simulate_human_decision(
            anomaly["anomaly_type"], anomaly["confidence"], result.action, is_tp, rng, anomaly["evidence"]
        )
        reward, _ = combine_reward(
            human_decision=human_decision,
            edit_distance=edit_dist,
            is_timeout_fallback=False,
            final_action=final_action,
            anomaly_type=anomaly["anomaly_type"],
            confidence=anomaly["confidence"],
            evidence=anomaly["evidence"],
            employee=employee,
            is_true_positive=is_tp,
            rng=rng,
        )
        store.add_incident(
            anomaly_id=anomaly["anomaly_id"],
            anomaly_type=anomaly["anomaly_type"],
            confidence=anomaly["confidence"],
            evidence=anomaly["evidence"],
            employee=employee,
            # the action the bandit/system actually chose (result.action),
            # not the human's correction -- same reasoning as train_cycles.py
            action_taken=result.action,
            is_true_positive=is_tp,
            is_timeout_fallback=False,
            human_decision=human_decision,
            reward=reward,
            collection_name=DEMO_COLLECTION,
        )
        return {"is_true_positive": is_tp, "human_decision": human_decision, "final_action": final_action, "reward": reward}

    def run_occurrence(anomaly: dict):
        context = context_vector(anomaly["anomaly_type"], anomaly["confidence"])
        employee = employees_by_id.get(anomaly["employee_id"])
        # query the demo's own isolated collection, not the shared one
        return select_action_with_memory(
            bandit, context, anomaly["anomaly_type"], anomaly["confidence"],
            anomaly["evidence"], employee, explore=False, collection_name=DEMO_COLLECTION,
        )

    print(f"\nbandit weights before either occurrence: update_count={bandit.update_count} (untrained, all-zero)")

    result_1 = run_occurrence(occurrence_1)
    outcome_1 = resolve_and_record(occurrence_1, result_1)

    result_2 = run_occurrence(occurrence_2)
    # deliberately NOT calling bandit.update() anywhere in this script --
    # the bandit's own weights must stay untouched between occurrences for
    # this to actually prove a memory effect, not a training effect

    print("\n=== Occurrence 1 (first time this anomaly type+context is seen) ===")
    print(f"chosen action:      {result_1.action}")
    print(f"confidence margin:  {result_1.margin:.3f}  (gap between best and 2nd-best scored action)")
    print(f"memory neighbors:   {result_1.neighbor_count}")
    print(f"used memory:        {result_1.used_memory}")
    print(f"resolved outcome:   true_positive={outcome_1['is_true_positive']}, "
          f"human={outcome_1['human_decision']}, reward={outcome_1['reward']:+.2f}")

    print("\n=== Occurrence 2 (a similar anomaly, same untrained bandit, memory now has 1 record) ===")
    print(f"chosen action:      {result_2.action}")
    print(f"confidence margin:  {result_2.margin:.3f}")
    print(f"memory neighbors:   {result_2.neighbor_count}")
    print(f"used memory:        {result_2.used_memory}")
    print(f"bandit update_count unchanged between occurrences: {bandit.update_count == 0}")

    print("\n=== The actual claim being proven ===")
    print(f"margin improved from {result_1.margin:.3f} to {result_2.margin:.3f} "
          f"({'higher confidence on 2nd occurrence -- PROVEN' if result_2.margin > result_1.margin else 'NOT proven this run'})")
    print(
        "the bandit's own weights never changed (0 training updates occurred) -- the entire "
        "improvement came from retrieving and reusing 1 stored memory, which is exactly what "
        "'warm-starting the RL policy from memory' means: skipping the slow part, not doing it faster."
    )


if __name__ == "__main__":
    main()
