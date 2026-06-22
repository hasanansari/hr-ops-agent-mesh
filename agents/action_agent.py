from agents.state import AgentName, HROpsState, TraceEntry


def action_agent_node(state: HROpsState) -> dict:
    """Stub. Real version executes a mock tool call (leave application,
    balance check, payslip fetch) with structured JSON I/O and retry logic."""
    node_input = {"raw_input": state.raw_input}
    result = {
        "tool_called": None,
        "status": "stub: tool execution not yet implemented",
    }
    # tool_calls is empty, not None, because this node ran -- it just had
    # nothing real to call yet. None would mean "this node doesn't track
    # tool calls"; [] means "it does, and there weren't any this time."
    trace_entry = TraceEntry(agent=AgentName.ACTION, input=node_input, output=result, tool_calls=[])
    return {"action_result": result, "trace": [trace_entry]}
