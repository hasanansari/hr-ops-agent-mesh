from agents.state import AgentName, HROpsState, TraceEntry


def anomaly_detection_node(state: HROpsState) -> dict:
    """Stub. Real version scans the employee dataset for payroll outliers,
    leave-abuse patterns, and compliance gaps, each with a confidence score
    and a recommended action."""
    node_input = {"trigger_type": state.trigger_type, "raw_input": state.raw_input}
    result = {
        "anomalies": [],
        "status": "stub: anomaly scan not yet implemented",
    }
    trace_entry = TraceEntry(agent=AgentName.ANOMALY_DETECTION, input=node_input, output=result)
    return {"anomaly_result": result, "trace": [trace_entry]}
