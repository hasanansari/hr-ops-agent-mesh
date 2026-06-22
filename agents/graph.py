import time

from langgraph.graph import END, StateGraph

from agents.action_agent import action_agent_node
from agents.anomaly_agent import anomaly_detection_node
from agents.bandit_agent import bandit_agent_node
from agents.compliance_agent import compliance_agent_node, compliance_veto_node, route_after_compliance_veto
from agents.hitl_agent import hitl_gate_node
from agents.policy_agent import policy_agent_node
from agents.state import AgentName, HROpsState
from agents.supervisor import route_after_supervisor, supervisor_node


def _timed(node_fn):
    """Stamps latency_ms onto every trace entry a node returns, without
    touching any individual node's code. One wrapper applied uniformly here
    beats editing 7 separate node files to each start their own timer --
    the same reasoning as the bandit's epsilon-greedy choice over LinUCB:
    the simpler mechanism that still gets the job done correctly.
    """

    def wrapped(state):
        start = time.perf_counter()
        result = node_fn(state)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        if result.get("trace"):
            result["trace"] = [entry.model_copy(update={"latency_ms": elapsed_ms}) for entry in result["trace"]]
        return result

    return wrapped


def build_graph():
    builder = StateGraph(HROpsState)

    builder.add_node(AgentName.SUPERVISOR.value, _timed(supervisor_node))
    builder.add_node(AgentName.POLICY.value, _timed(policy_agent_node))
    builder.add_node(AgentName.ACTION.value, _timed(action_agent_node))
    builder.add_node(AgentName.ANOMALY_DETECTION.value, _timed(anomaly_detection_node))
    builder.add_node(AgentName.COMPLIANCE.value, _timed(compliance_agent_node))
    builder.add_node(AgentName.HITL_GATE.value, _timed(hitl_gate_node))
    builder.add_node(AgentName.BANDIT.value, _timed(bandit_agent_node))
    builder.add_node(AgentName.COMPLIANCE_VETO.value, _timed(compliance_veto_node))

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
