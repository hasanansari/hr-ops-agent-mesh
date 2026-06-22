from pathlib import Path

from agents.state import AgentName, HROpsState, TraceEntry
from bandit.policy import LinearEpsilonGreedyBandit, context_vector

POLICY_PATH = Path(__file__).resolve().parent.parent / "bandit" / "policy_state.json"


def bandit_agent_node(state: HROpsState) -> dict:
    """Attaches the bandit's action recommendation alongside Section B's
    rule-based one on every anomaly -- augmenting, not replacing it. The
    rule-based recommendation still drives HITL/Action Agent behaviour
    downstream; the bandit's suggestion rides along for comparison until
    there's enough validated performance history to trust it to override
    the rule outright. Inference only -- explore=False, no exploration in
    the live path, that belongs in the offline training cycles
    (bandit/train_cycles.py), not in a real scan a human is about to review.
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

    all_anomalies = anomaly_result["high_confidence_anomalies"] + anomaly_result["review_queue"]
    agree_count = 0
    for anomaly in all_anomalies:
        context = context_vector(anomaly["anomaly_type"], anomaly["confidence"])
        bandit_action = bandit.select_action(context, explore=False)
        anomaly["bandit_action"] = bandit_action
        if bandit_action == anomaly["recommended_action"]:
            agree_count += 1

    node_input = {"anomaly_count": len(all_anomalies), "policy_update_count": bandit.update_count}
    trace_output = {
        "agreement_with_rule_based": f"{agree_count}/{len(all_anomalies)}" if all_anomalies else "0/0",
    }
    trace_entry = TraceEntry(agent=AgentName.BANDIT, input=node_input, output=trace_output)

    return {
        "anomaly_result": anomaly_result,
        "bandit_result": {"agree_count": agree_count, "total": len(all_anomalies)},
        "trace": [trace_entry],
    }
