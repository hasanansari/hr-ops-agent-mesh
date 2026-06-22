"""LLM cost optimization analysis: what a naive all-LLM version of one
scheduled-scan run would have cost, vs what this system actually spends.

Pricing is Claude Sonnet 4.6's real published rate: $3.00 / 1M input tokens,
$15.00 / 1M output tokens.

Token counts here are an approximation (~4 characters/token), NOT the exact
Anthropic tokenizer -- getting an exact count requires a live call to
`/v1/messages/count_tokens`, and this analysis deliberately makes zero live
API calls (consistent with every other "no live LLM calls" choice in this
project). The 4-chars/token rule is the standard rough estimate for English
text and is accurate to within ~15-20% -- precise enough to compare two
architectures' order of magnitude, which is the actual question here, not
billing-grade precision.

The call counts and the text fed into each prompt are real, though: pulled
from an actual Section B scan over the actual 800-employee dataset, not
invented numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.anomaly_scoring import run_anomaly_scan
from compliance.rules_engine import RULES_PATH
from data.generate_employees import generate_employees

CHARS_PER_TOKEN = 4.0  # the standard rough English-text approximation

# Real Claude Sonnet 4.6 pricing, per current Anthropic published rates.
INPUT_PRICE_PER_MILLION = 3.00
OUTPUT_PRICE_PER_MILLION = 15.00


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def call_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION + (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION


# ---------------------------------------------------------------------------
# Naive-baseline prompt construction. "Naive" means: no statistics, no
# rules engine, no learned policy -- every decision point in the pipeline
# instead makes one full-context LLM call. Each function below builds the
# prompt a naive implementation would realistically send, using real data
# from this project, not invented placeholder text.
# ---------------------------------------------------------------------------


def supervisor_prompt(raw_input: str, trigger_type: str) -> tuple[str, str]:
    system = (
        "You are the triage supervisor for an HR Ops platform. Given a trigger type "
        "(reactive_query, scheduled_scan, system_alert) and raw input text, decide which "
        "specialist should handle it: policy_agent, action_agent, anomaly_detection_agent, "
        "or compliance_agent. Respond with just the agent name."
    )
    user = f"trigger_type: {trigger_type}\nraw_input: {raw_input}"
    return system, user


def per_employee_check_prompt(employee: dict, cohort: list[dict]) -> tuple[str, str]:
    system = (
        "You are an HR anomaly detector. Given one employee's full record and their full "
        "peer cohort (same department and level), determine whether their payroll, leave, "
        "or compliance data is anomalous relative to the cohort. Report is_anomalous (bool), "
        "anomaly_type, confidence (0-1), and your reasoning."
    )
    user = f"employee: {json.dumps(employee)}\npeer_cohort ({len(cohort)} employees): {json.dumps(cohort)}"
    return system, user


def action_selection_prompt(anomaly_type: str, confidence: float, evidence: dict, employee: dict) -> tuple[str, str]:
    system = (
        "You are an HR Ops action-selection agent. Given a detected anomaly's type, "
        "confidence, evidence, and the employee's record, choose exactly one action: "
        "auto-correct, escalate-to-manager, escalate-to-HR, flag-for-audit, or no-action. "
        "Explain your reasoning in 1-2 sentences."
    )
    user = (
        f"anomaly_type: {anomaly_type}\nconfidence: {confidence}\n"
        f"evidence: {json.dumps(evidence)}\nemployee: {json.dumps(employee)}"
    )
    return system, user


def compliance_check_prompt(
    rules_text: str, anomaly_type: str, confidence: float, evidence: dict, employee: dict, proposed_action: str
) -> tuple[str, str]:
    system = (
        "You are an HR compliance officer. Evaluate the proposed action against this "
        f"ruleset:\n{rules_text}\nIf any rule is violated, return the safest compliant "
        "replacement action and which rules were violated. Otherwise confirm the action "
        "is compliant."
    )
    user = (
        f"anomaly_type: {anomaly_type}\nconfidence: {confidence}\nevidence: {json.dumps(evidence)}\n"
        f"employee: {json.dumps(employee)}\nproposed_action: {proposed_action}"
    )
    return system, user


# Representative output text per call type -- a real LLM call also bills
# for what it generates, not just what it's given. These are hand-written
# but realistic example responses, counted with the same estimator, not
# assumed to be free.
REPRESENTATIVE_OUTPUTS = {
    "supervisor": "anomaly_detection_agent",
    "per_employee_check": (
        '{"is_anomalous": true, "anomaly_type": "payroll_outlier", "confidence": 0.92, '
        '"reasoning": "Salary is more than 4 standard deviations above the peer cohort '
        'mean for this department and level, which is far outside the normal spread '
        'observed across the other employees in the same cohort."}'
    ),
    "action_selection": (
        "escalate-to-HR. The payroll discrepancy is large relative to peers and the "
        "detector confidence is high, so this needs HR sign-off rather than an "
        "automated correction or a lighter-touch escalation."
    ),
    "compliance_check": (
        "Violation: PAYROLL_CORRECTION_TIER_LARGE. The discrepancy exceeds the $5,000 "
        "threshold that requires HR sign-off, so flag-for-audit is not sufficient here. "
        "Recommended replacement action: escalate-to-HR."
    ),
}


def run_analysis() -> dict:
    employees, _ = generate_employees()
    scan = run_anomaly_scan(employees)
    anomalies = scan["high_confidence_anomalies"] + scan["review_queue"]
    hitl_reviewed = scan["high_confidence_anomalies"]  # these are the ones that reach the compliance gate

    employees_by_id = {e["employee_id"]: e for e in employees}
    employees_by_cohort: dict[tuple, list[dict]] = {}
    for e in employees:
        employees_by_cohort.setdefault((e["department"], e["level"]), []).append(e)

    rules_text = RULES_PATH.read_text()

    out_supervisor = estimate_tokens(REPRESENTATIVE_OUTPUTS["supervisor"])
    out_employee_check = estimate_tokens(REPRESENTATIVE_OUTPUTS["per_employee_check"])
    out_action = estimate_tokens(REPRESENTATIVE_OUTPUTS["action_selection"])
    out_compliance = estimate_tokens(REPRESENTATIVE_OUTPUTS["compliance_check"])

    # --- Call type 1: supervisor routing (1 call/request) ---
    sys_text, user_text = supervisor_prompt(
        "nightly cycle: scan attendance and payroll for anomalies", "scheduled_scan"
    )
    supervisor_in = estimate_tokens(sys_text + user_text)
    supervisor_calls = 1
    supervisor_cost = call_cost(supervisor_in, out_supervisor) * supervisor_calls

    # --- Call type 2: per-employee anomaly check (1 call/employee) ---
    sample_employee = employees[0]
    sample_cohort = employees_by_cohort[(sample_employee["department"], sample_employee["level"])]
    sys_text, user_text = per_employee_check_prompt(sample_employee, sample_cohort)
    employee_check_in = estimate_tokens(sys_text + user_text)
    employee_check_calls = len(employees)
    employee_check_cost = call_cost(employee_check_in, out_employee_check) * employee_check_calls

    # --- Call type 3: bandit-equivalent action selection (1 call/detected anomaly) ---
    sample_anomaly = anomalies[0]
    sample_anomaly_employee = employees_by_id[sample_anomaly["employee_id"]]
    sys_text, user_text = action_selection_prompt(
        sample_anomaly["anomaly_type"], sample_anomaly["confidence"], sample_anomaly["evidence"], sample_anomaly_employee
    )
    action_in = estimate_tokens(sys_text + user_text)
    action_calls = len(anomalies)
    action_cost = call_cost(action_in, out_action) * action_calls

    # --- Call type 4: compliance veto check (1 call/HITL-reviewed decision) ---
    sample_hitl = hitl_reviewed[0]
    sample_hitl_employee = employees_by_id[sample_hitl["employee_id"]]
    sys_text, user_text = compliance_check_prompt(
        rules_text,
        sample_hitl["anomaly_type"],
        sample_hitl["confidence"],
        sample_hitl["evidence"],
        sample_hitl_employee,
        sample_hitl["recommended_action"],
    )
    compliance_in = estimate_tokens(sys_text + user_text)
    compliance_calls = len(hitl_reviewed)
    compliance_cost = call_cost(compliance_in, out_compliance) * compliance_calls

    naive_total_cost = supervisor_cost + employee_check_cost + action_cost + compliance_cost
    naive_total_tokens = (
        supervisor_calls * (supervisor_in + out_supervisor)
        + employee_check_calls * (employee_check_in + out_employee_check)
        + action_calls * (action_in + out_action)
        + compliance_calls * (compliance_in + out_compliance)
    )

    # The actual system, today: every one of these four decision points is
    # statistics (z-score), a linear bandit, or a YAML rules engine -- zero
    # LLM calls, by architectural choice, not by omission.
    actual_total_cost = 0.0
    actual_total_tokens = 0

    return {
        "call_types": [
            {
                "name": "Supervisor routing",
                "calls": supervisor_calls,
                "input_tokens_per_call": supervisor_in,
                "output_tokens_per_call": out_supervisor,
                "total_cost": round(supervisor_cost, 6),
            },
            {
                "name": "Per-employee anomaly check (naive equivalent of Section B's z-score)",
                "calls": employee_check_calls,
                "input_tokens_per_call": employee_check_in,
                "output_tokens_per_call": out_employee_check,
                "total_cost": round(employee_check_cost, 4),
            },
            {
                "name": "Action selection (naive equivalent of Section C's bandit)",
                "calls": action_calls,
                "input_tokens_per_call": action_in,
                "output_tokens_per_call": out_action,
                "total_cost": round(action_cost, 4),
            },
            {
                "name": "Compliance veto check (naive equivalent of Section E's rules engine)",
                "calls": compliance_calls,
                "input_tokens_per_call": compliance_in,
                "output_tokens_per_call": out_compliance,
                "total_cost": round(compliance_cost, 4),
            },
        ],
        "naive_total_tokens": naive_total_tokens,
        "naive_total_cost": round(naive_total_cost, 4),
        "actual_total_tokens": actual_total_tokens,
        "actual_total_cost": actual_total_cost,
        "reduction_pct": 100.0 if naive_total_cost > 0 else 0.0,
        "scan_size": {"employees": len(employees), "anomalies": len(anomalies), "hitl_reviewed": len(hitl_reviewed)},
    }


def print_report(analysis: dict) -> None:
    print("=== LLM Cost Optimization Analysis ===")
    print(f"(based on a real scan over {analysis['scan_size']['employees']} employees, "
          f"{analysis['scan_size']['anomalies']} detected anomalies, "
          f"{analysis['scan_size']['hitl_reviewed']} HITL-reviewed)")
    print(f"Pricing: Claude Sonnet 4.6 -- ${INPUT_PRICE_PER_MILLION:.2f}/1M input, "
          f"${OUTPUT_PRICE_PER_MILLION:.2f}/1M output tokens\n")

    print(f"{'Call type':<60}{'calls':>8}{'tok/call':>10}{'cost ($)':>12}")
    for ct in analysis["call_types"]:
        tok_per_call = ct["input_tokens_per_call"] + ct["output_tokens_per_call"]
        print(f"{ct['name']:<60}{ct['calls']:>8}{tok_per_call:>10}{ct['total_cost']:>12.4f}")

    print(f"\n{'NAIVE BASELINE TOTAL':<60}{'':>8}{'':>10}{analysis['naive_total_cost']:>12.4f}")
    print(f"{'  total tokens (naive)':<60}{analysis['naive_total_tokens']:>28,}")
    print(f"\n{'ACTUAL SYSTEM TOTAL':<60}{'':>8}{'':>10}{analysis['actual_total_cost']:>12.4f}")
    print(f"{'  total tokens (actual)':<60}{analysis['actual_total_tokens']:>28,}")

    print(f"\nReduction: {analysis['reduction_pct']:.0f}% (required: >=20%)")
    print(
        "\nWhy 100%, not some smaller optimized number: this isn't a cost optimization\n"
        "applied AFTER an LLM-based design -- it's the architectural decision (Sections\n"
        "B/C/E) to not reach for an LLM at all for these four decision types, since\n"
        "z-score statistics, a linear bandit, and declarative YAML rules answer the same\n"
        "questions an LLM would, deterministically, in microseconds, for zero tokens.\n"
        "The one place a real system would legitimately need an LLM is natural-language\n"
        "policy Q&A (Policy Agent's RAG flow) -- that's still a Section A stub, so it\n"
        "contributes $0 to both sides of this comparison today. Once built, that's the\n"
        "one path where the optimization is retrieval shrinking the context, not\n"
        "eliminating the LLM call -- natural language understanding is a task an LLM is\n"
        "actually needed for, unlike the four compared here."
    )


if __name__ == "__main__":
    analysis = run_analysis()
    print_report(analysis)

    out_path = Path(__file__).parent / "cost_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nfull analysis written to {out_path}")
