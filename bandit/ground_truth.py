"""Shared true-positive lookup, used by both the simulated cycles and the
real-HITL-data ingestion path so they agree on what "actually happened."
Relies on data/generate_employees.py's fixed SEED=42 default -- the
employee_id -> ground truth mapping is reproducible across any run, past
or present, as long as nobody changes that seed.
"""

from __future__ import annotations


def ground_truth_lookup(truth: list[dict]) -> dict[str, dict]:
    return {row["employee_id"]: row for row in truth}


def is_true_positive(anomaly_type: str, evidence: dict, employee_id: str, ground_truth_by_id: dict[str, dict]) -> bool:
    row = ground_truth_by_id[employee_id]
    if anomaly_type == "payroll_outlier":
        return bool(row["is_payroll_outlier"])
    if anomaly_type == "leave_abuse":
        return bool(row["is_leave_abuser"])
    violation = evidence.get("violation")
    if violation == "overtime_cap_breach":
        return bool(row["is_overtime_breach"])
    if violation == "missing_mandatory_training":
        return bool(row["is_missing_training"])
    raise ValueError(f"unrecognized anomaly_type/evidence: {anomaly_type} {evidence}")
