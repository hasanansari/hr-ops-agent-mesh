from pathlib import Path

from agents.anomaly_scoring import load_employees, run_anomaly_scan
from agents.state import AgentName, HROpsState, TraceEntry

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.csv"


def anomaly_detection_node(state: HROpsState) -> dict:
    """Runs the full payroll/leave/compliance scan over the employee
    dataset. The scan is dataset-wide regardless of trigger type for now --
    filtering a reactive query down to e.g. "just Engineering" would need
    NL parsing that doesn't exist yet, so that's a known limitation.
    """
    employees = load_employees(DATASET_PATH)
    result = run_anomaly_scan(employees)

    node_input = {
        "trigger_type": state.trigger_type,
        "raw_input": state.raw_input,
        "scanned_count": result["scanned_count"],
    }
    # the trace gets a summary, not the full anomaly dump -- the full list
    # lives in `anomaly_result` for anything downstream that needs detail.
    trace_output = {
        "anomaly_count": result["anomaly_count"],
        "high_confidence_count": len(result["high_confidence_anomalies"]),
        "pending_review_count": len(result["review_queue"]),
    }
    trace_entry = TraceEntry(agent=AgentName.ANOMALY_DETECTION, input=node_input, output=trace_output)
    return {"anomaly_result": result, "trace": [trace_entry]}
