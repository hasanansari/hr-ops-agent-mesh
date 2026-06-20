from langgraph.graph import END, StateGraph

from agents.action_agent import action_agent_node
from agents.anomaly_agent import anomaly_detection_node
from agents.compliance_agent import compliance_agent_node
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
        AgentName.ANOMALY_DETECTION,
        AgentName.COMPLIANCE,
    ):
        builder.add_edge(agent.value, END)

    return builder.compile()


graph = build_graph()
