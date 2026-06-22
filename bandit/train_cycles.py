"""The core deliverable: run 2 feedback cycles over the real Section B
dataset and show the bandit's recommendations measurably shift between
them. No matplotlib in this project's dependencies, so the "plot" is a
clean printed table + ASCII bars -- explicitly sanctioned as the
lower-effort alternative in the brief, and it's still screenshot-able.

Each cycle is the same loop: detect anomalies -> bandit recommends an
action per anomaly -> a (simulated) human reviews it -> combine_reward
scores that outcome -> the bandit updates its weights from that reward.
The policy is reset fresh at the start of this script on purpose, so the
before/after comparison is reproducible from a clean state every time this
file is run -- cross-run persistence is proven separately in
test_persistence.py, this script's job is to prove *learning*, not
restart-safety.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.anomaly_models import RecommendedAction
from agents.anomaly_scoring import run_anomaly_scan
from bandit.ground_truth import ground_truth_lookup, is_true_positive
from bandit.hitl_feedback import train_on_real_decisions
from bandit.policy import ACTIONS, LinearEpsilonGreedyBandit, context_vector
from bandit.reward import combine_reward
from bandit.simulate_human import ideal_rank, simulate_human_decision
from data.generate_employees import generate_employees
from hitl.models import ACTION_RANK
from memory import store as memory_store

POLICY_PATH = Path(__file__).parent / "policy_state.json"
RESULTS_PATH = Path(__file__).parent / "cycle_results.json"

SIMULATED_TIMEOUT_RATE = 0.05  # exercises the "timeout isn't a signal" exclusion path live


def _is_true_positive(anomaly: dict, ground_truth_by_id: dict[str, dict]) -> bool:
    return is_true_positive(anomaly["anomaly_type"], anomaly["evidence"], anomaly["employee_id"], ground_truth_by_id)


def run_cycle(
    bandit: LinearEpsilonGreedyBandit,
    anomalies: list[dict],
    ground_truth_by_id: dict[str, dict],
    employees_by_id: dict[str, dict],
    rng: np.random.Generator,
    cycle_label: str,
) -> list[dict]:
    log = []
    for anomaly in anomalies:
        context = context_vector(anomaly["anomaly_type"], anomaly["confidence"])
        chosen_action = bandit.select_action(context, explore=True)
        is_tp = _is_true_positive(anomaly, ground_truth_by_id)
        is_timeout = rng.random() < SIMULATED_TIMEOUT_RATE

        if is_timeout:
            human_decision, final_action, edit_dist = None, chosen_action, None
        else:
            human_decision, final_action, edit_dist = simulate_human_decision(
                anomaly["anomaly_type"], anomaly["confidence"], chosen_action, is_tp, rng, anomaly["evidence"]
            )

        reward, breakdown = combine_reward(
            human_decision=human_decision,
            edit_distance=edit_dist,
            is_timeout_fallback=is_timeout,
            final_action=final_action,
            anomaly_type=anomaly["anomaly_type"],
            confidence=anomaly["confidence"],
            evidence=anomaly["evidence"],
            employee=employees_by_id.get(anomaly["employee_id"]),
            is_true_positive=is_tp,
            rng=rng,
        )
        bandit.update(context, chosen_action, reward)

        # Section F: this is the one place context+action+outcome+reward
        # are all genuinely known together, so it's the natural place to
        # persist the resolution as an episodic memory -- written to the
        # same collection the live graph reads from in bandit_agent_node,
        # so real training here actually warm-starts real future scans.
        #
        # action_taken is chosen_action, NOT final_action -- the reward
        # just computed is credit/blame for the action the *system*
        # picked (same as what bandit.update() just used above), not for
        # whatever a human corrected it to. Memory has to bias future
        # proposals against the same action the reward is actually about,
        # or it silently points the wrong direction.
        employee = employees_by_id.get(anomaly["employee_id"])
        memory_store.add_incident(
            anomaly_id=anomaly["anomaly_id"],
            anomaly_type=anomaly["anomaly_type"],
            confidence=anomaly["confidence"],
            evidence=anomaly["evidence"],
            employee=employee,
            action_taken=chosen_action,
            is_true_positive=is_tp,
            is_timeout_fallback=is_timeout,
            human_decision=human_decision,
            reward=reward,
        )

        log.append(
            {
                "cycle": cycle_label,
                "anomaly_id": anomaly["anomaly_id"],
                "anomaly_type": anomaly["anomaly_type"],
                "confidence": anomaly["confidence"],
                "chosen_action": chosen_action,
                "is_true_positive": is_tp,
                "is_timeout_fallback": is_timeout,
                "human_decision": human_decision,
                "final_action": final_action,
                "reward": reward,
                "reward_breakdown": breakdown,
                "triggered_compliance_veto": breakdown["compliance_veto"] != 0.0,
                "rank_gap": abs(
                    ACTION_RANK[RecommendedAction(chosen_action)]
                    - ideal_rank(anomaly["anomaly_type"], is_tp, anomaly["evidence"])
                ),
            }
        )
    return log


def _action_distribution(log: list[dict]) -> dict[str, int]:
    counts = {a: 0 for a in ACTIONS}
    for row in log:
        counts[row["chosen_action"]] += 1
    return counts


def _ascii_bar(count: int, max_count: int, width: int = 30) -> str:
    filled = round(width * count / max_count) if max_count else 0
    return "#" * filled


def print_report(log1: list[dict], log2: list[dict]) -> None:
    dist1, dist2 = _action_distribution(log1), _action_distribution(log2)
    max_count = max(max(dist1.values()), max(dist2.values()), 1)

    print("\n=== Action distribution: cycle 1 (cold start) vs cycle 2 (after learning) ===")
    print(f"{'action':<22}{'cycle 1':>9}  {'':<32}{'cycle 2':>9}")
    for action in ACTIONS:
        bar1 = _ascii_bar(dist1[action], max_count)
        bar2 = _ascii_bar(dist2[action], max_count)
        print(f"{action:<22}{dist1[action]:>9}  {bar1:<32}{dist2[action]:>9}  {bar2}")

    reward1 = sum(r["reward"] for r in log1)
    reward2 = sum(r["reward"] for r in log2)
    avg_gap1 = sum(r["rank_gap"] for r in log1) / len(log1)
    avg_gap2 = sum(r["rank_gap"] for r in log2) / len(log2)
    vetoes1 = sum(1 for r in log1 if r["triggered_compliance_veto"])
    vetoes2 = sum(1 for r in log2 if r["triggered_compliance_veto"])

    print("\n=== Cumulative reward ===")
    print(f"cycle 1 total reward: {reward1:+.2f}  (avg {reward1 / len(log1):+.3f} per decision over {len(log1)} decisions)")
    print(f"cycle 2 total reward: {reward2:+.2f}  (avg {reward2 / len(log2):+.3f} per decision over {len(log2)} decisions)")

    print("\n=== Section E: compliance vetoes triggered by the bandit's own chosen action ===")
    print(f"cycle 1: {vetoes1}/{len(log1)} decisions vetoed (each costing an extra {-1.0:+.1f} reward)")
    print(f"cycle 2: {vetoes2}/{len(log2)} decisions vetoed")
    print(f"-> {'fewer vetoes after learning' if vetoes2 < vetoes1 else 'no improvement in veto rate'}")

    print("\n=== Average distance from the 'ideal' action rank (0 = perfect) ===")
    print(f"cycle 1: {avg_gap1:.3f}")
    print(f"cycle 2: {avg_gap2:.3f}")
    print(f"-> {'improved' if avg_gap2 < avg_gap1 else 'did not improve'} by {avg_gap1 - avg_gap2:+.3f}")

    print("\n=== Cumulative reward curve (checkpoints every 20 decisions) ===")
    combined = log1 + log2
    running = 0.0
    print(f"{'decision #':>10}  {'cumulative reward':>18}  cycle")
    for i, row in enumerate(combined, start=1):
        running += row["reward"]
        if i % 20 == 0 or i == len(combined):
            print(f"{i:>10}  {running:>18.2f}  {row['cycle']}")


def main() -> None:
    employees, truth = generate_employees()
    ground_truth_by_id = ground_truth_lookup(truth)
    employees_by_id = {e["employee_id"]: e for e in employees}

    scan = run_anomaly_scan(employees)
    anomalies = scan["high_confidence_anomalies"] + scan["review_queue"]
    print(f"running both cycles over the same {len(anomalies)} detected anomalies from the real Section B scan")

    bandit = LinearEpsilonGreedyBandit(epsilon=0.15, learning_rate=0.1, seed=1)

    # real Section D wiring: ingest whatever has actually been decided
    # through hitl/app.py so far, before any simulated training happens.
    real_rng = np.random.default_rng(99)
    real_log = train_on_real_decisions(bandit, real_rng)
    real_timeouts = sum(1 for r in real_log if r["is_timeout_fallback"])
    print(
        f"ingested {len(real_log)} real HITL decisions from hitl/decisions.sqlite "
        f"({real_timeouts} were timeout fallbacks, contributing 0 reward each by design)"
    )

    rng1 = np.random.default_rng(101)
    log1 = run_cycle(bandit, anomalies, ground_truth_by_id, employees_by_id, rng1, cycle_label="cycle_1")

    bandit.save(POLICY_PATH)
    bandit = LinearEpsilonGreedyBandit.load(POLICY_PATH, seed=2)  # genuinely reload from disk before cycle 2

    rng2 = np.random.default_rng(202)
    log2 = run_cycle(bandit, anomalies, ground_truth_by_id, employees_by_id, rng2, cycle_label="cycle_2")

    bandit.save(POLICY_PATH)
    print_report(log1, log2)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"cycle_1": log1, "cycle_2": log2}, f, indent=2)
    print(f"\nfull per-decision log written to {RESULTS_PATH}")
    print(f"trained policy persisted to {POLICY_PATH}")


if __name__ == "__main__":
    main()
