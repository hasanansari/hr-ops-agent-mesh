"""Mock implementations of the 3 tool schemas in schemas.py. Each one reads
real records from the same employee dataset every other section already
uses (data/employees.csv), rather than inventing a separate mock data
source. Each call has a simulated chance of a transient failure -- a real
upstream HR system call can time out or 5xx -- so the retry wrapper below
has something real to retry against, not just untested code.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from agents.anomaly_scoring import load_employees

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.csv"

TRANSIENT_FAILURE_RATE = 0.3
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.2


class ToolUnavailableError(Exception):
    """Simulates a transient upstream failure (timeout, 5xx) -- the kind a
    retry is actually supposed to recover from, as opposed to a real
    application error (e.g. unknown employee_id), which retrying can't fix
    and shouldn't try to."""


def _employee_by_id(employee_id: str) -> dict | None:
    employees = load_employees(DATASET_PATH)
    return next((e for e in employees if e["employee_id"] == employee_id), None)


def _maybe_fail(rng: random.Random) -> None:
    if rng.random() < TRANSIENT_FAILURE_RATE:
        raise ToolUnavailableError("upstream HR system timed out")


def check_leave_balance(employee_id: str, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    _maybe_fail(rng)
    employee = _employee_by_id(employee_id)
    if employee is None:
        return {"status": "error", "error": f"unknown employee_id: {employee_id}"}
    return {
        "status": "ok",
        "employee_id": employee_id,
        "leave_balance_days": employee["leave_balance_days"],
        "leave_taken_qtr": employee["leave_taken_qtr"],
    }


def apply_for_leave(employee_id: str, days: int, start_date: str, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    _maybe_fail(rng)
    employee = _employee_by_id(employee_id)
    if employee is None:
        return {"status": "error", "error": f"unknown employee_id: {employee_id}"}
    if days > employee["leave_balance_days"]:
        # a real application-level rejection, not a transient failure --
        # retrying this would never help, so it's returned, not raised.
        return {
            "status": "rejected",
            "reason": f"requested {days} days exceeds remaining balance of {employee['leave_balance_days']}",
        }
    return {
        "status": "ok",
        "employee_id": employee_id,
        "days_approved": days,
        "start_date": start_date,
        "remaining_balance_after": employee["leave_balance_days"] - days,
    }


def fetch_payslip(employee_id: str, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    _maybe_fail(rng)
    employee = _employee_by_id(employee_id)
    if employee is None:
        return {"status": "error", "error": f"unknown employee_id: {employee_id}"}
    return {
        "status": "ok",
        "employee_id": employee_id,
        "period": "current",
        "gross_pay": employee["monthly_salary"],
    }


TOOL_IMPLEMENTATIONS = {
    "check_leave_balance": check_leave_balance,
    "apply_for_leave": apply_for_leave,
    "fetch_payslip": fetch_payslip,
}


def call_tool_with_retry(tool_name: str, tool_input: dict, rng: random.Random | None = None) -> dict:
    """Retries only on ToolUnavailableError (transient) -- an application
    error like "rejected" or "unknown employee_id" comes back as a normal
    result on the first attempt and is never retried, since trying again
    wouldn't change the outcome.
    """
    rng = rng or random.Random()
    tool_fn = TOOL_IMPLEMENTATIONS[tool_name]
    attempts = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = tool_fn(rng=rng, **tool_input)
            attempts.append({"attempt": attempt, "outcome": "succeeded"})
            return {"tool_name": tool_name, "input": tool_input, "result": result, "attempts": attempts}
        except ToolUnavailableError as e:
            attempts.append({"attempt": attempt, "outcome": "transient_failure", "error": str(e)})
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)  # linear backoff

    # every retry exhausted -- a meaningful fallback, not a crash
    return {
        "tool_name": tool_name,
        "input": tool_input,
        "result": {"status": "unavailable", "error": "upstream HR system unreachable after retries"},
        "attempts": attempts,
    }
