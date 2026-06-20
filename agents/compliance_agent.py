from agents.state import AgentName, HROpsState, TraceEntry


def compliance_agent_node(state: HROpsState) -> dict:
    """Stub. Real version evaluates a proposed action against the YAML/JSON
    rules engine and issues a hard veto on violations, overriding any
    upstream recommendation."""
    node_input = {"trigger_type": state.trigger_type, "raw_input": state.raw_input}
    result = {
        "veto": False,
        "violated_rules": [],
        "status": "stub: compliance rules engine not yet implemented",
    }
    trace_entry = TraceEntry(agent=AgentName.COMPLIANCE, input=node_input, output=result)
    return {"compliance_result": result, "trace": [trace_entry]}
