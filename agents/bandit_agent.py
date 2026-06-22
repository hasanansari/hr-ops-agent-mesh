from pathlib import Path

from agents.anomaly_scoring import load_employees
from agents.state import AgentName, HROpsState, TraceEntry
from bandit.policy import LinearEpsilonGreedyBandit, context_vector
from memory.warm_start import select_action_with_memory

POLICY_PATH = Path(__file__).resolve().parent.parent / "bandit" / "policy_state.json"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.csv"


def bandit_agent_node(state: HROpsState) -> dict:
    """Attaches the bandit's action recommendation alongside Section B's
    rule-based one on every anomaly -- augmenting, not replacing it. The
    rule-based recommendation still drives HITL/Action Agent behaviour
    downstream; the bandit's suggestion rides along for comparison until
    there's enough validated performance history to trust it to override
    the rule outright.

    The recommendation itself is "warm-started" from episodic memory
    (Section F): before trusting its own (possibly still-untrained)
    weights, it checks whether similar past incidents are on record and
    biases toward whatever action worked well for them. Inference only --
    explore=False, no exploration in the live path; that belongs in the
    offline training cycles (bandit/train_cycles.py), not in a real scan a
    human is about to review.
    """
    anomaly_result = state.anomaly_result
    if not anomaly_result:
        trace_entry = TraceEntry(
            agent=AgentName.BANDIT,
            input={"anomaly_count": 0},
            output={"status": "nothing to score"},
        )
        return {"trace": [trace_entry]}

    bandit = LinearEpsilonGreedyBandit.load_or_new(POLICY_PATH, epsilon=0.15, learning_rate=0.1)
    employees_by_id = {e["employee_id"]: e for e in load_employees(DATASET_PATH)}

    all_anomalies = anomaly_result["high_confidence_anomalies"] + anomaly_result["review_queue"]
    agree_count = 0
    memory_assisted_count = 0
    for anomaly in all_anomalies:
        context = context_vector(anomaly["anomaly_type"], anomaly["confidence"])
        employee = employees_by_id.get(anomaly["employee_id"])
        result = select_action_with_memory(
            bandit,
            context,
            anomaly["anomaly_type"],
            anomaly["confidence"],
            anomaly["evidence"],
            employee,
            explore=False,
        )
        anomaly["bandit_action"] = result.action
        anomaly["bandit_confidence_margin"] = round(result.margin, 3)
        anomaly["bandit_memory_neighbors"] = result.neighbor_count
        if result.used_memory:
            memory_assisted_count += 1
        if result.action == anomaly["recommended_action"]:
            agree_count += 1

    node_input = {"anomaly_count": len(all_anomalies), "policy_update_count": bandit.update_count}
    trace_output = {
        "agreement_with_rule_based": f"{agree_count}/{len(all_anomalies)}" if all_anomalies else "0/0",
        "memory_assisted": f"{memory_assisted_count}/{len(all_anomalies)}" if all_anomalies else "0/0",
    }
    trace_entry = TraceEntry(agent=AgentName.BANDIT, input=node_input, output=trace_output)

    return {
        "anomaly_result": anomaly_result,
        "bandit_result": {
            "agree_count": agree_count,
            "total": len(all_anomalies),
            "memory_assisted_count": memory_assisted_count,
        },
        "trace": [trace_entry],
    }
