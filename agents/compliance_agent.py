import json
from pathlib import Path

from agents.anomaly_models import RecommendedAction
from agents.anomaly_scoring import load_employees
from agents.state import AgentName, HROpsState, TraceEntry
from compliance.rules_engine import evaluate, load_rules

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.csv"


def compliance_agent_node(state: HROpsState) -> dict:
    """Handles reactive/system-alert compliance queries routed here directly
    by the Supervisor (e.g. "payroll engine alert: overtime cap breach...").
    The rules engine itself is real (see compliance_veto_node below) -- what
    is NOT built is turning raw alert/query text into the structured
    (anomaly_type, evidence, employee) the engine needs to evaluate. That's
    an NLP-parsing problem, not a rules-engine problem, and is a known,
    named limitation rather than a silent gap: this node reports the engine
    is loaded and ready, but can't yet act on free text alone.
    """
    rules = load_rules()
    node_input = {"trigger_type": state.trigger_type, "raw_input": state.raw_input}
    result = {
        "veto": False,
        "violated_rules": [],
        "status": f"rules engine loaded ({len(rules)} rules) -- alert/query text parsing not implemented",
    }
    trace_entry = TraceEntry(agent=AgentName.COMPLIANCE, input=node_input, output=result)
    return {"compliance_result": result, "trace": [trace_entry]}


def compliance_veto_node(state: HROpsState) -> dict:
    """The real veto gate: runs after the HITL gate, before the Action
    Agent, and re-checks every decided anomaly's final_action against the
    YAML ruleset -- including overriding an explicit human decision when a
    hard rule demands it (see compliance/rules_engine.py's evaluate() for
    why that's deliberate, not a bug). Anything the engine overrides is
    surfaced in compliance_veto_result; the override -- not the original
    HITL decision -- is what the Action Agent actually receives. The
    original human decision in hitl/decisions.sqlite is left untouched, so
    the audit trail still shows what a human actually said.
    """
    hitl_result = state.hitl_result or {}
    reviewed = hitl_result.get("reviewed", [])

    if not reviewed:
        trace_entry = TraceEntry(
            agent=AgentName.COMPLIANCE_VETO,
            input={"reviewed_count": 0},
            output={"status": "nothing to evaluate"},
        )
        return {"compliance_veto_result": {"overrides": [], "actionable": []}, "trace": [trace_entry]}

    rules = load_rules()
    employees_by_id = {e["employee_id"]: e for e in load_employees(DATASET_PATH)}

    overrides = []
    actionable = []
    for decision in reviewed:
        employee = employees_by_id.get(decision["employee_id"])
        # the HITL store keeps evidence as a JSON string column, not a
        # parsed dict -- has to be decoded before the engine can read it.
        evidence = json.loads(decision["evidence_json"]) if decision.get("evidence_json") else {}
        verdict = evaluate(
            anomaly_type=decision["anomaly_type"],
            confidence=decision["confidence"],
            evidence=evidence,
            final_action=decision["final_action"],
            employee=employee,
            rules=rules,
        )
        if verdict.veto:
            overrides.append(
                {
                    "anomaly_id": decision["anomaly_id"],
                    "employee_id": decision["employee_id"],
                    "original_action": verdict.original_action,
                    "overridden_action": verdict.final_action,
                    "violated_rule_ids": verdict.violated_rule_ids,
                }
            )
        final_action = verdict.final_action
        if final_action != RecommendedAction.NO_ACTION.value:
            actionable.append(decision["anomaly_id"])

    node_input = {"reviewed_count": len(reviewed)}
    trace_output = {"veto_count": len(overrides), "actionable_count": len(actionable)}
    trace_entry = TraceEntry(agent=AgentName.COMPLIANCE_VETO, input=node_input, output=trace_output)

    return {
        "compliance_veto_result": {"overrides": overrides, "actionable": actionable},
        "trace": [trace_entry],
    }


def route_after_compliance_veto(state: HROpsState) -> str:
    veto_result = state.compliance_veto_result or {}
    if veto_result.get("actionable"):
        return AgentName.ACTION.value
    return "end"
