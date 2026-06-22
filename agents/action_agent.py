import re

from agents.state import AgentName, HROpsState, TraceEntry
from tools.mock_tools import call_tool_with_retry

# No identity/session system exists yet (no auth, no "who is asking"
# context on a reactive query) -- a real system would resolve this from
# the request; this demo path uses a fixed employee for self-service tool
# calls, same kind of named simplification as the rest of this project.
DEMO_EMPLOYEE_ID = "EMP-0001"

_DAYS_RE = re.compile(r"(\d+)\s*day")


def _select_tool(raw_input: str) -> tuple[str, dict] | None:
    """Cheap keyword match, same style as supervisor.py's triage -- real
    parameter extraction (parsing "June 15" into an ISO date, etc.) would
    need NLP/structured extraction; out of scope here, so a fixed
    placeholder date is used when one can't be trivially parsed.
    """
    text = raw_input.lower()
    if "balance" in text:
        return "check_leave_balance", {"employee_id": DEMO_EMPLOYEE_ID}
    if "payslip" in text or "pay slip" in text or "salary slip" in text:
        return "fetch_payslip", {"employee_id": DEMO_EMPLOYEE_ID}
    if "apply" in text and "leave" in text:
        days_match = _DAYS_RE.search(text)
        days = int(days_match.group(1)) if days_match else 1
        return "apply_for_leave", {"employee_id": DEMO_EMPLOYEE_ID, "days": days, "start_date": "2026-07-01"}
    return None


def action_agent_node(state: HROpsState) -> dict:
    """Real tool execution for the reactive self-service path (check leave
    balance / apply for leave / fetch payslip), each call going through a
    retry-wrapped mock tool (tools/mock_tools.py) with structured JSON I/O.

    Also handles the post-compliance-veto remediation path: when this node
    is reached after a scheduled scan, it's reporting what final_action got
    executed per anomaly, not calling a self-service tool -- escalating or
    auto-correcting an anomaly isn't the same kind of action as an
    employee checking their own balance, so it's reported as its own
    structured result rather than forced through the same 3 tool schemas.
    """
    node_input = {"raw_input": state.raw_input}

    veto_result = state.compliance_veto_result
    if veto_result is not None:
        actionable_ids = veto_result.get("actionable", [])
        result = {
            "status": "ok" if actionable_ids else "nothing_actionable",
            "executed_count": len(actionable_ids),
            "anomaly_ids": actionable_ids,
        }
        trace_entry = TraceEntry(agent=AgentName.ACTION, input=node_input, output=result, tool_calls=[])
        return {"action_result": result, "trace": [trace_entry]}

    selected = _select_tool(state.raw_input)
    if selected is None:
        result = {"status": "no_matching_tool", "tool_called": None}
        trace_entry = TraceEntry(agent=AgentName.ACTION, input=node_input, output=result, tool_calls=[])
        return {"action_result": result, "trace": [trace_entry]}

    tool_name, tool_input = selected
    call_record = call_tool_with_retry(tool_name, tool_input)

    result = {"status": "ok", "tool_called": tool_name, "tool_result": call_record["result"]}
    trace_entry = TraceEntry(agent=AgentName.ACTION, input=node_input, output=result, tool_calls=[call_record])
    return {"action_result": result, "trace": [trace_entry]}
