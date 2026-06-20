from agents.state import AgentName, HROpsState, TraceEntry


def policy_agent_node(state: HROpsState) -> dict:
    """Stub. Real version retrieves chunks from the embedded HR policy doc
    and grounds an answer in them (RAG)."""
    node_input = {"raw_input": state.raw_input}
    result = {
        "answer": "stub: policy lookup not yet implemented",
        "citations": [],
    }
    trace_entry = TraceEntry(agent=AgentName.POLICY, input=node_input, output=result)
    return {"policy_result": result, "trace": [trace_entry]}
