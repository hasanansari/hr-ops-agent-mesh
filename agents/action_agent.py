from agents.state import AgentName, HROpsState, TraceEntry


def action_agent_node(state: HROpsState) -> dict:
    """Stub. Real version executes a mock tool call (leave application,
    balance check, payslip fetch) with structured JSON I/O and retry logic."""
    node_input = {"raw_input": state.raw_input}
    result = {
        "tool_called": None,
        "status": "stub: tool execution not yet implemented",
    }
    trace_entry = TraceEntry(agent=AgentName.ACTION, input=node_input, output=result)
    return {"action_result": result, "trace": [trace_entry]}
