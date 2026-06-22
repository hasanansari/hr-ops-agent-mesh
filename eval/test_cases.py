"""15-test evaluation harness: happy path (5), edge cases (4), adversarial
inputs (3), RL-specific (3). Each test is a small function returning
(passed: bool, reasoning: str) -- the reasoning is what actually gets
reported, not just a checkmark, since "why did this pass/fail" is the
useful part of an eval report.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

# Short HITL window for every test in this file -- these are correctness
# checks, not a HITL demo; nothing here needs to sit through the 2-minute
# production default.
os.environ.setdefault("HITL_TIMEOUT_SECONDS", "3")
os.environ.setdefault("HITL_POLL_INTERVAL_SECONDS", "1")

from agents.anomaly_scoring import run_anomaly_scan
from agents.compliance_agent import compliance_veto_node, route_after_compliance_veto
from agents.graph import graph
from agents.hitl_agent import hitl_gate_node
from agents.state import AgentName, HROpsState, TriggerType
from bandit.policy import ACTIONS, LinearEpsilonGreedyBandit, context_vector
from bandit.reward import combine_reward
from compliance.rules_engine import evaluate as evaluate_compliance

RESULTS_PATH = Path(__file__).parent / "eval_results.json"


# ---------------------------------------------------------------------------
# Happy path (5) -- normal reactive queries, scheduled scans, system alerts
# routing correctly. These are the cases the system handles every day.
# ---------------------------------------------------------------------------


def test_h1_reactive_query_to_policy():
    state = HROpsState(trigger_type=TriggerType.REACTIVE_QUERY, raw_input="What is our maternity leave policy?")
    result = graph.invoke(state)
    passed = result["route"] == AgentName.POLICY
    return passed, f"a generic policy question should route to Policy; got route={result['route']}"


def test_h2_reactive_query_to_action():
    state = HROpsState(trigger_type=TriggerType.REACTIVE_QUERY, raw_input="Apply for 3 days leave starting June 15")
    result = graph.invoke(state)
    passed = result["route"] == AgentName.ACTION
    return passed, f"an 'apply for leave' request should route to Action; got route={result['route']}"


def test_h3_reactive_query_to_anomaly_detection():
    state = HROpsState(
        trigger_type=TriggerType.REACTIVE_QUERY,
        raw_input="Flag anyone in Engineering who has taken more than 15 days leave in Q1",
    )
    result = graph.invoke(state)
    passed = result["route"] == AgentName.ANOMALY_DETECTION
    return passed, f"a 'flag anyone' request should match the anomaly keyword list; got route={result['route']}"


def test_h4_scheduled_scan_produces_results():
    state = HROpsState(trigger_type=TriggerType.SCHEDULED_SCAN, raw_input="nightly cycle")
    result = graph.invoke(state)
    routed_correctly = result["route"] == AgentName.ANOMALY_DETECTION
    has_results = result["anomaly_result"] is not None and result["anomaly_result"]["anomaly_count"] > 0
    passed = routed_correctly and has_results
    return passed, (
        f"scheduled scans always route to AnomalyDetection and should find >0 anomalies in "
        f"the real 800-employee dataset; route={result['route']}, "
        f"anomaly_count={result['anomaly_result']['anomaly_count'] if result['anomaly_result'] else None}"
    )


def test_h5_system_alert_to_compliance():
    state = HROpsState(
        trigger_type=TriggerType.SYSTEM_ALERT,
        raw_input="payroll engine alert: overtime cap breach detected for EMP-4821",
    )
    result = graph.invoke(state)
    passed = result["route"] == AgentName.COMPLIANCE
    return passed, f"system alerts always route to Compliance; got route={result['route']}"


# ---------------------------------------------------------------------------
# Edge cases (4) -- the boundaries where the normal flow shouldn't apply.
# ---------------------------------------------------------------------------


def test_e1_empty_dataset():
    result = run_anomaly_scan([])
    passed = (
        result["scanned_count"] == 0
        and result["anomaly_count"] == 0
        and result["high_confidence_anomalies"] == []
        and result["review_queue"] == []
    )
    return passed, f"scanning zero employees should return empty results, not crash; got {result}"


def test_e2_all_anomalies_below_threshold():
    # simulates a scan where nothing crossed the high-confidence bar --
    # the gate should have nothing to review and route straight to END,
    # not block waiting for a human or call the Action Agent.
    state = HROpsState(trigger_type=TriggerType.SCHEDULED_SCAN, raw_input="scan")
    state.anomaly_result = {"high_confidence_anomalies": [], "review_queue": [{"fake": "low confidence item"}]}

    hitl_update = hitl_gate_node(state)
    state.hitl_result = hitl_update["hitl_result"]

    veto_update = compliance_veto_node(state)
    state.compliance_veto_result = veto_update["compliance_veto_result"]

    next_route = route_after_compliance_veto(state)
    passed = (
        hitl_update["hitl_result"]["reviewed"] == []
        and veto_update["compliance_veto_result"]["actionable"] == []
        and next_route == "end"
    )
    return passed, (
        f"with nothing above the confidence threshold, the gate should have 0 reviewed items "
        f"and route to END, not the Action Agent; got next_route={next_route!r}"
    )


def test_e3_all_hitl_timeouts():
    # nobody touches hitl/app.py during this test, so with a 3-second
    # window every high-confidence item should fall back, not get treated
    # as a real human decision.
    state = HROpsState(trigger_type=TriggerType.SCHEDULED_SCAN, raw_input="nightly cycle")
    result = graph.invoke(state)
    reviewed = (result.get("hitl_result") or {}).get("reviewed", [])
    passed = len(reviewed) > 0 and all(
        d["is_timeout_fallback"] and d["final_action"] == "flag-for-audit" and d["human_decision"] is None
        for d in reviewed
    )
    return passed, (
        f"with no human reviewer running, every item should time out to flag-for-audit with "
        f"human_decision=None, never silently counted as approval; reviewed_count={len(reviewed)}"
    )


def test_e4_compliance_veto_overrides_human_approval():
    # a human approves a $7,000 payroll discrepancy resolved with only
    # flag-for-audit -- PAYROLL_CORRECTION_TIER_LARGE demands escalate-to-HR
    # regardless of who signed off.
    verdict = evaluate_compliance(
        anomaly_type="payroll_outlier",
        confidence=0.95,
        evidence={"salary": 16000, "cohort_mean": 9000},
        final_action="flag-for-audit",
        employee=None,
    )
    passed = verdict.veto is True and verdict.final_action == "escalate-to-HR"
    return passed, (
        f"a hard compliance rule must override even an explicit human approval, not just the "
        f"rule-based/RL recommendation; veto={verdict.veto}, overridden_to={verdict.final_action}"
    )


# ---------------------------------------------------------------------------
# Adversarial inputs (3) -- inputs designed to break something.
# ---------------------------------------------------------------------------


def test_a1_malformed_empty_input():
    state = HROpsState(trigger_type=TriggerType.REACTIVE_QUERY, raw_input="")
    try:
        result = graph.invoke(state)
        passed = result["route"] is not None
        return passed, f"empty raw_input should still resolve to a route (the policy default), not crash; got route={result['route']}"
    except Exception as e:
        return False, f"empty raw_input crashed the graph: {e!r}"


def test_a2_unknown_trigger_type():
    try:
        HROpsState(trigger_type="not_a_real_trigger_type", raw_input="test")
        return False, "an invalid trigger_type should be rejected at construction, but no exception was raised"
    except Exception as e:
        return True, f"correctly rejected an unrecognized trigger_type before it could silently misroute: {type(e).__name__}"


def test_a3_missing_required_field():
    try:
        HROpsState(trigger_type=TriggerType.REACTIVE_QUERY)  # raw_input omitted
        return False, "constructing state without raw_input should fail validation, but no exception was raised"
    except Exception as e:
        return True, f"correctly rejected state missing a required field: {type(e).__name__}"


# ---------------------------------------------------------------------------
# RL-specific (3)
# ---------------------------------------------------------------------------


def test_r1_bandit_cold_start():
    bandit = LinearEpsilonGreedyBandit(epsilon=0.0, seed=1)
    context = context_vector("payroll_outlier", 0.9)
    action = bandit.select_action(context, explore=False)
    passed = action in ACTIONS and bandit.update_count == 0
    return passed, (
        f"a brand-new bandit (0 updates, all-zero weights) should still return a valid action "
        f"via random tie-breaking, not crash or default to a fixed choice; action={action!r}"
    )


def test_r2_persistence_survives_restart():
    path = Path(tempfile.mktemp(suffix=".json"))
    try:
        bandit = LinearEpsilonGreedyBandit(epsilon=0.0, learning_rate=0.2, seed=1)
        context = context_vector("leave_abuse", 0.8)
        for _ in range(10):
            bandit.update(context, "escalate-to-manager", 1.0)
        bandit.save(path)

        reloaded = LinearEpsilonGreedyBandit.load(path)  # a fresh object, simulating a restart
        action_before = bandit.select_action(context, explore=False)
        action_after = reloaded.select_action(context, explore=False)
        passed = action_before == action_after and reloaded.update_count == bandit.update_count
        return passed, (
            f"a reloaded policy must select the same action as before saving; "
            f"before={action_before!r} after={action_after!r} "
            f"update_count before={bandit.update_count} after={reloaded.update_count}"
        )
    finally:
        path.unlink(missing_ok=True)


def test_r3_compliance_penalty_reduces_reward():
    rng = np.random.default_rng(0)
    reward_with_veto, breakdown_with = combine_reward(
        human_decision="approve",
        edit_distance=None,
        is_timeout_fallback=False,
        final_action="flag-for-audit",
        anomaly_type="payroll_outlier",
        confidence=0.95,
        evidence={"salary": 16000, "cohort_mean": 9000},  # $7,000 gap -> triggers PAYROLL_CORRECTION_TIER_LARGE
        employee=None,
        is_true_positive=True,
        rng=rng,
    )
    reward_without_veto, breakdown_without = combine_reward(
        human_decision="approve",
        edit_distance=None,
        is_timeout_fallback=False,
        final_action="escalate-to-manager",
        anomaly_type="payroll_outlier",
        confidence=0.95,
        evidence={"salary": 9200, "cohort_mean": 9000},  # $200 gap -> compliant
        employee=None,
        is_true_positive=True,
        rng=rng,
    )
    passed = reward_with_veto < reward_without_veto and breakdown_with["compliance_veto"] < 0
    return passed, (
        f"an approved action that still triggers a compliance veto must score lower than an "
        f"equally-approved, compliant one; with_veto={reward_with_veto:+.2f} "
        f"without_veto={reward_without_veto:+.2f} veto_penalty={breakdown_with['compliance_veto']:+.2f}"
    )


TESTS = [
    ("H1", "happy_path", "Reactive query routes to Policy", test_h1_reactive_query_to_policy),
    ("H2", "happy_path", "Reactive query routes to Action", test_h2_reactive_query_to_action),
    ("H3", "happy_path", "Reactive query routes to AnomalyDetection", test_h3_reactive_query_to_anomaly_detection),
    ("H4", "happy_path", "Scheduled scan produces real results", test_h4_scheduled_scan_produces_results),
    ("H5", "happy_path", "System alert routes to Compliance", test_h5_system_alert_to_compliance),
    ("E1", "edge_case", "Empty dataset scans without crashing", test_e1_empty_dataset),
    ("E2", "edge_case", "All anomalies below threshold -> nothing to review", test_e2_all_anomalies_below_threshold),
    ("E3", "edge_case", "All HITL timeouts fall back correctly", test_e3_all_hitl_timeouts),
    ("E4", "edge_case", "Compliance veto overrides human approval", test_e4_compliance_veto_overrides_human_approval),
    ("A1", "adversarial", "Malformed/empty input doesn't crash", test_a1_malformed_empty_input),
    ("A2", "adversarial", "Unknown trigger_type is rejected", test_a2_unknown_trigger_type),
    ("A3", "adversarial", "Missing required field is rejected", test_a3_missing_required_field),
    ("R1", "rl_specific", "Bandit cold-start behavior is valid", test_r1_bandit_cold_start),
    ("R2", "rl_specific", "Persistence survives a restart", test_r2_persistence_survives_restart),
    ("R3", "rl_specific", "Compliance penalty reduces reward", test_r3_compliance_penalty_reduces_reward),
]


def run_eval() -> list[dict]:
    results = []
    for test_id, category, description, fn in TESTS:
        try:
            passed, reasoning = fn()
        except Exception as e:
            passed, reasoning = False, f"test raised an unhandled exception: {e!r}"
        results.append(
            {"id": test_id, "category": category, "description": description, "passed": passed, "reasoning": reasoning}
        )
        print(f"[{'PASS' if passed else 'FAIL'}] {test_id} ({category}) {description}")
        print(f"       {reasoning}")
    return results


def print_summary(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{passed}/{total} tests passed")
    for category in ("happy_path", "edge_case", "adversarial", "rl_specific"):
        cat_results = [r for r in results if r["category"] == category]
        cat_passed = sum(1 for r in cat_results if r["passed"])
        print(f"  {category:<14} {cat_passed}/{len(cat_results)}")


if __name__ == "__main__":
    results = run_eval()
    print_summary(results)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull eval report written to {RESULTS_PATH}")
