from langgraph.graph import END, StateGraph

from agents.action_agent import action_agent_node
from agents.anomaly_agent import anomaly_detection_node
from agents.compliance_agent import compliance_agent_node
from agents.hitl_agent import hitl_gate_node, route_after_hitl_gate
from agents.policy_agent import policy_agent_node
from agents.state import AgentName, HROpsState
from agents.supervisor import route_after_supervisor, supervisor_node


def build_graph():
    builder = StateGraph(HROpsState)

    builder.add_node(AgentName.SUPERVISOR.value, supervisor_node)
    builder.add_node(AgentName.POLICY.value, policy_agent_node)
    builder.add_node(AgentName.ACTION.value, action_agent_node)
    builder.add_node(AgentName.ANOMALY_DETECTION.value, anomaly_detection_node)
    builder.add_node(AgentName.COMPLIANCE.value, compliance_agent_node)
    builder.add_node(AgentName.HITL_GATE.value, hitl_gate_node)

    builder.set_entry_point(AgentName.SUPERVISOR.value)

    # Conditional edge: the supervisor doesn't call a sub-agent directly, it
    # writes `route` to state and this mapping decides who runs next.
    builder.add_conditional_edges(
        AgentName.SUPERVISOR.value,
        route_after_supervisor,
        {
            AgentName.POLICY.value: AgentName.POLICY.value,
            AgentName.ACTION.value: AgentName.ACTION.value,
            AgentName.ANOMALY_DETECTION.value: AgentName.ANOMALY_DETECTION.value,
            AgentName.COMPLIANCE.value: AgentName.COMPLIANCE.value,
        },
    )

    for agent in (
        AgentName.POLICY,
        AgentName.ACTION,
        AgentName.COMPLIANCE,
    ):
        builder.add_edge(agent.value, END)

    # AnomalyDetection no longer dead-ends at END -- every scan result now
    # passes through the HITL gate before anything gets a chance to act.
    builder.add_edge(AgentName.ANOMALY_DETECTION.value, AgentName.HITL_GATE.value)

    # The gate doesn't call the Action Agent directly -- it writes
    # hitl_result to state, and this conditional edge reads it to decide
    # whether anything approved/modified/fallen-back-to actually needs the
    # Action Agent, or whether the run is done.
    builder.add_conditional_edges(
        AgentName.HITL_GATE.value,
        route_after_hitl_gate,
        {
            AgentName.ACTION.value: AgentName.ACTION.value,
            "end": END,
        },
    )

    return builder.compile()


graph = build_graph()
