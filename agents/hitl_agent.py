import time

from agents.anomaly_models import RecommendedAction
from agents.state import AgentName, HROpsState, TraceEntry
from hitl import store
from hitl.models import DEFAULT_TIMEOUT_ACTION


def hitl_gate_node(state: HROpsState) -> dict:
    """Blocks until every high-confidence anomaly from this scan has been
    decided by a human (via the Streamlit app, in a separate process writing
    to the same SQLite store) or the configurable timeout window elapses.

    This is a deliberate simplification worth naming: blocking a thread for
    up to the timeout window is fine for a prototype, but holds a process
    hostage in production. The production-grade version would use
    LangGraph's interrupt()/checkpointer mechanism to fully suspend the run
    and resume it later via a webhook, instead of polling in a sleep loop.
    """
    anomaly_result = state.anomaly_result or {}
    high_confidence = anomaly_result.get("high_confidence_anomalies", [])

    if not high_confidence:
        trace_entry = TraceEntry(
            agent=AgentName.HITL_GATE,
            input={"high_confidence_count": 0},
            output={"status": "nothing to review"},
        )
        return {"hitl_result": {"reviewed": [], "approved_for_action": []}, "trace": [trace_entry]}

    for anomaly in high_confidence:
        store.create_pending(anomaly, state.request_id)

    anomaly_ids = [a["anomaly_id"] for a in high_confidence]
    timeout_seconds = store.get_timeout_seconds()
    poll_interval = store.get_poll_interval_seconds()
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if all(store.get_status(aid) != "pending" for aid in anomaly_ids):
            break
        time.sleep(poll_interval)

    for aid in anomaly_ids:
        if store.get_status(aid) == "pending":
            store.mark_timeout_fallback(aid, DEFAULT_TIMEOUT_ACTION.value)

    decisions = [store.get_decision(aid) for aid in anomaly_ids]
    approved_for_action = [
        d["anomaly_id"] for d in decisions if d["final_action"] and d["final_action"] != RecommendedAction.NO_ACTION.value
    ]

    node_input = {"trigger_type": state.trigger_type, "high_confidence_count": len(high_confidence)}
    trace_output = {
        "approved_count": sum(1 for d in decisions if d["human_decision"] == "approve"),
        "rejected_count": sum(1 for d in decisions if d["human_decision"] == "reject"),
        "modified_count": sum(1 for d in decisions if d["human_decision"] == "modify"),
        "timeout_fallback_count": sum(1 for d in decisions if d["is_timeout_fallback"]),
        "approved_for_action_count": len(approved_for_action),
    }
    trace_entry = TraceEntry(agent=AgentName.HITL_GATE, input=node_input, output=trace_output)

    return {
        "hitl_result": {"reviewed": decisions, "approved_for_action": approved_for_action},
        "trace": [trace_entry],
    }
