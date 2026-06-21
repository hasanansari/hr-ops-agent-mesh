"""Anomaly detectors for the HR Ops platform.

Deliberately statistics-based, not model-based -- every score here is a
formula you can compute on paper from numbers visible in `evidence`. That's
the point: confidence has to be defensible to a human reviewer, not a
black-box probability.

Two different kinds of detector live here, and they score confidence
differently on purpose:

  - Payroll outliers and leave abuse are *inferred* from a distribution --
    "this looks unusual compared to peers" could always be a false
    positive (a legitimately senior hire, a real bereavement leave), so
    confidence is graded continuously off how extreme the statistic is.
  - Compliance violations (missing training, overtime cap breach) are
    *directly observed* facts against a hard rule -- there's no
    inference step, so confidence starts high and only varies with
    severity, not with "how sure are we this happened."
"""

from collections import defaultdict
from csv import DictReader
from pathlib import Path

import numpy as np

from agents.anomaly_models import Anomaly, AnomalyType, RecommendedAction, ReviewStatus

# A z-score of 2 is the conventional ~95%-confidence-interval cutoff under a
# normal distribution -- the standard "this is statistically unusual"
# threshold. 4 is treated as the point past which we're maximally sure;
# confidence is linearly interpolated between the two.
Z_FLAG_THRESHOLD = 2.0
Z_SATURATION = 4.0

# Below this many peers, a cohort's mean/std is too noisy to trust (a
# single outlier can swing a 3-person cohort's std wildly) -- skip rather
# than risk a confident-looking false positive.
MIN_COHORT_SIZE = 5

# Anomalies at or above this confidence are trusted enough to flag for
# automatic action; below it, they queue for human review instead. Set
# conservatively high (rather than e.g. 0.5) because two of the three
# detectors are statistical inferences that can be wrong -- the cost of a
# wrongly-automated payroll or leave action is much higher than the cost
# of an extra item in a human's review queue.
HIGH_CONFIDENCE_THRESHOLD = 0.8

# Matches the org's own stated leave policy example ("more than 15 days
# leave in Q1").
LEAVE_POLICY_LIMIT_QTR_DAYS = 15
LEAVE_OVER_LIMIT_WEIGHT = 0.6
LEAVE_CLUSTERING_WEIGHT = 0.4
LEAVE_CLUSTERING_FLAG_RATIO = 0.5

# Mock policy constant. A real system would pull this from the YAML
# compliance rules engine (Section E) instead of hardcoding it here -- it's
# inlined for now since that engine doesn't exist yet.
OVERTIME_CAP_HOURS_PER_WEEK = 12.0


def load_employees(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = DictReader(f)
        return [
            {
                "employee_id": row["employee_id"],
                "name": row["name"],
                "department": row["department"],
                "level": int(row["level"]),
                "title": row["title"],
                "tenure_months": int(row["tenure_months"]),
                "monthly_salary": float(row["monthly_salary"]),
                "overtime_hours_week": float(row["overtime_hours_week"]),
                "mandatory_training_completed": row["mandatory_training_completed"] == "True",
                "leave_taken_qtr": int(row["leave_taken_qtr"]),
                "leave_long_weekend_count": int(row["leave_long_weekend_count"]),
                "leave_balance_days": int(row["leave_balance_days"]),
                "performance_rating": int(row["performance_rating"]),
                "attendance_rate": float(row["attendance_rate"]),
            }
            for row in reader
        ]


def _scale(value: float, lo: float, hi: float) -> float:
    """Linear rescale of value from [lo, hi] to [0, 1], clipped at both ends."""
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


def _status(confidence: float) -> ReviewStatus:
    return ReviewStatus.AUTO_TRIGGERED if confidence >= HIGH_CONFIDENCE_THRESHOLD else ReviewStatus.PENDING_HUMAN_REVIEW


def detect_payroll_outliers(employees: list[dict]) -> list[Anomaly]:
    """Peer cohort = same (department, level). Flags anyone whose salary is
    >= 2 std deviations from their cohort's mean.
    """
    cohorts: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for emp in employees:
        cohorts[(emp["department"], emp["level"])].append(emp)

    anomalies = []
    for (department, level), members in cohorts.items():
        if len(members) < MIN_COHORT_SIZE:
            continue
        salaries = np.array([m["monthly_salary"] for m in members])
        mean, std = float(salaries.mean()), float(salaries.std(ddof=1))
        if std == 0:
            continue

        for emp in members:
            z = (emp["monthly_salary"] - mean) / std
            if abs(z) < Z_FLAG_THRESHOLD:
                continue
            confidence = _scale(abs(z), Z_FLAG_THRESHOLD, Z_SATURATION)
            if confidence >= 0.85:
                action = RecommendedAction.ESCALATE_TO_HR
            elif confidence >= 0.6:
                action = RecommendedAction.ESCALATE_TO_MANAGER
            elif confidence >= 0.35:
                action = RecommendedAction.FLAG_FOR_AUDIT
            else:
                action = RecommendedAction.NO_ACTION

            anomalies.append(
                Anomaly(
                    employee_id=emp["employee_id"],
                    anomaly_type=AnomalyType.PAYROLL_OUTLIER,
                    confidence=round(confidence, 3),
                    recommended_action=action,
                    status=_status(confidence),
                    requires_action_agent=confidence >= HIGH_CONFIDENCE_THRESHOLD,
                    evidence={
                        "department": department,
                        "level": level,
                        "cohort_size": len(members),
                        "salary": emp["monthly_salary"],
                        "cohort_mean": round(mean, 2),
                        "cohort_std": round(std, 2),
                        "z_score": round(z, 2),
                    },
                )
            )
    return anomalies


def detect_leave_abuse(employees: list[dict]) -> list[Anomaly]:
    """Two signals, blended: how far over the quarterly policy limit (60%
    weight) and what fraction of leave days are single Mon/Fri days
    clustered around a weekend -- the classic long-weekend-gaming pattern
    (40% weight). Either signal alone can trigger a flag; the blend just
    sets the confidence once flagged.
    """
    anomalies = []
    for emp in employees:
        taken = emp["leave_taken_qtr"]
        clustered = emp["leave_long_weekend_count"]
        over_limit_ratio = max(0.0, (taken - LEAVE_POLICY_LIMIT_QTR_DAYS) / LEAVE_POLICY_LIMIT_QTR_DAYS)
        clustering_ratio = clustered / taken if taken else 0.0

        if taken <= LEAVE_POLICY_LIMIT_QTR_DAYS and clustering_ratio <= LEAVE_CLUSTERING_FLAG_RATIO:
            continue

        confidence = min(
            1.0,
            LEAVE_OVER_LIMIT_WEIGHT * min(over_limit_ratio, 1.0) + LEAVE_CLUSTERING_WEIGHT * clustering_ratio,
        )
        if confidence >= 0.85:
            action = RecommendedAction.ESCALATE_TO_MANAGER
        elif confidence >= 0.5:
            action = RecommendedAction.FLAG_FOR_AUDIT
        else:
            action = RecommendedAction.NO_ACTION

        anomalies.append(
            Anomaly(
                employee_id=emp["employee_id"],
                anomaly_type=AnomalyType.LEAVE_ABUSE,
                confidence=round(confidence, 3),
                recommended_action=action,
                status=_status(confidence),
                requires_action_agent=confidence >= HIGH_CONFIDENCE_THRESHOLD,
                evidence={
                    "leave_taken_qtr": taken,
                    "policy_limit_qtr": LEAVE_POLICY_LIMIT_QTR_DAYS,
                    "long_weekend_count": clustered,
                    "clustering_ratio": round(clustering_ratio, 2),
                },
            )
        )
    return anomalies


def detect_compliance_violations(employees: list[dict]) -> list[Anomaly]:
    """Two hard rule checks, not statistical inference -- confidence here
    means "how severe," not "how sure."
    """
    anomalies = []
    for emp in employees:
        if not emp["mandatory_training_completed"]:
            confidence = 0.9  # directly observed boolean fact, near-certain
            anomalies.append(
                Anomaly(
                    employee_id=emp["employee_id"],
                    anomaly_type=AnomalyType.COMPLIANCE_VIOLATION,
                    confidence=confidence,
                    recommended_action=RecommendedAction.ESCALATE_TO_MANAGER,
                    status=_status(confidence),
                    requires_action_agent=confidence >= HIGH_CONFIDENCE_THRESHOLD,
                    evidence={"violation": "missing_mandatory_training"},
                )
            )

        overtime = emp["overtime_hours_week"]
        if overtime > OVERTIME_CAP_HOURS_PER_WEEK:
            severity = (overtime - OVERTIME_CAP_HOURS_PER_WEEK) / OVERTIME_CAP_HOURS_PER_WEEK
            confidence = min(1.0, 0.7 + 0.3 * min(severity, 1.0))
            action = RecommendedAction.AUTO_CORRECT if severity < 0.5 else RecommendedAction.ESCALATE_TO_HR
            anomalies.append(
                Anomaly(
                    employee_id=emp["employee_id"],
                    anomaly_type=AnomalyType.COMPLIANCE_VIOLATION,
                    confidence=round(confidence, 3),
                    recommended_action=action,
                    status=_status(confidence),
                    requires_action_agent=confidence >= HIGH_CONFIDENCE_THRESHOLD,
                    evidence={
                        "violation": "overtime_cap_breach",
                        "overtime_hours_week": overtime,
                        "cap": OVERTIME_CAP_HOURS_PER_WEEK,
                        "severity": round(severity, 2),
                    },
                )
            )
    return anomalies


def run_anomaly_scan(employees: list[dict]) -> dict:
    anomalies = [
        *detect_payroll_outliers(employees),
        *detect_leave_abuse(employees),
        *detect_compliance_violations(employees),
    ]
    high_confidence = [a for a in anomalies if a.requires_action_agent]
    review_queue = [a for a in anomalies if not a.requires_action_agent]
    return {
        "scanned_count": len(employees),
        "anomaly_count": len(anomalies),
        "high_confidence_anomalies": [a.model_dump() for a in high_confidence],
        "review_queue": [a.model_dump() for a in review_queue],
    }
