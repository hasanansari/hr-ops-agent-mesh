from agents.state import AgentName, HROpsState, TraceEntry, TriggerType

_ACTION_KEYWORDS = ("apply", "leave application", "approve", "payslip", "request")
_ANOMALY_KEYWORDS = ("flag anyone", "flag all", "scan", "outlier", "pattern")
_COMPLIANCE_KEYWORDS = ("violat", "overtime cap", "compliance", "policy breach")


def _triage(state: HROpsState) -> AgentName:
    """Stub triage logic. Trigger type alone decides scheduled/system-generated
    requests; reactive natural-language queries get a cheap keyword classifier
    for now -- this is the seam where an LLM-based router slots in later
    without touching any graph wiring.
    """
    if state.trigger_type == TriggerType.SCHEDULED_SCAN:
        return AgentName.ANOMALY_DETECTION
    if state.trigger_type == TriggerType.SYSTEM_ALERT:
        return AgentName.COMPLIANCE

    text = state.raw_input.lower()
    if any(kw in text for kw in _COMPLIANCE_KEYWORDS):
        return AgentName.COMPLIANCE
    if any(kw in text for kw in _ANOMALY_KEYWORDS):
        return AgentName.ANOMALY_DETECTION
    if any(kw in text for kw in _ACTION_KEYWORDS):
        return AgentName.ACTION
    return AgentName.POLICY


def supervisor_node(state: HROpsState) -> dict:
    decision = _triage(state)
    trace_entry = TraceEntry(
        agent=AgentName.SUPERVISOR,
        input={"trigger_type": state.trigger_type, "raw_input": state.raw_input},
        output={"route": decision},
    )
    return {"route": decision, "trace": [trace_entry]}


def route_after_supervisor(state: HROpsState) -> str:
    """Conditional-edge function: reads the decision the supervisor node
    wrote to state and tells the graph runtime which node to run next.
    Routing logic lives here, not inside agent code, so no agent needs to
    know who else exists.
    """
    return state.route.value
