from langgraph.graph import END, StateGraph

from agents.action_agent import action_agent_node
from agents.anomaly_agent import anomaly_detection_node
from agents.bandit_agent import bandit_agent_node
from agents.compliance_agent import compliance_agent_node, compliance_veto_node, route_after_compliance_veto
from agents.hitl_agent import hitl_gate_node
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
    builder.add_node(AgentName.BANDIT.value, bandit_agent_node)
    builder.add_node(AgentName.COMPLIANCE_VETO.value, compliance_veto_node)

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
    # passes through the bandit (attaches a learned-policy suggestion
    # alongside each anomaly's rule-based one) and then the HITL gate
    # before anything gets a chance to act.
    builder.add_edge(AgentName.ANOMALY_DETECTION.value, AgentName.BANDIT.value)
    builder.add_edge(AgentName.BANDIT.value, AgentName.HITL_GATE.value)

    # The HITL gate's decisions are not final -- every one of them, even an
    # explicit human approval or rejection, still passes through the
    # compliance veto gate before anything is allowed to reach the Action
    # Agent. This is the literal "even if the Supervisor or RL policy
    # recommends the action" hard-veto requirement, extended to humans too.
    builder.add_edge(AgentName.HITL_GATE.value, AgentName.COMPLIANCE_VETO.value)

    builder.add_conditional_edges(
        AgentName.COMPLIANCE_VETO.value,
        route_after_compliance_veto,
        {
            AgentName.ACTION.value: AgentName.ACTION.value,
            "end": END,
        },
    )

    return builder.compile()


graph = build_graph()
