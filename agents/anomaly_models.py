from enum import Enum
from typing import Any

from pydantic import BaseModel


class AnomalyType(str, Enum):
    PAYROLL_OUTLIER = "payroll_outlier"
    LEAVE_ABUSE = "leave_abuse"
    COMPLIANCE_VIOLATION = "compliance_violation"


class RecommendedAction(str, Enum):
    AUTO_CORRECT = "auto-correct"
    ESCALATE_TO_MANAGER = "escalate-to-manager"
    ESCALATE_TO_HR = "escalate-to-HR"
    FLAG_FOR_AUDIT = "flag-for-audit"
    NO_ACTION = "no-action"


class ReviewStatus(str, Enum):
    AUTO_TRIGGERED = "auto_triggered"
    PENDING_HUMAN_REVIEW = "pending_human_review"


class Anomaly(BaseModel):
    employee_id: str
    anomaly_type: AnomalyType
    confidence: float
    recommended_action: RecommendedAction
    status: ReviewStatus
    requires_action_agent: bool
    # the numbers that produced the score, kept alongside the verdict so a
    # human reviewer (or a future LLM explainer) can see exactly why this
    # was flagged without re-running the detector.
    evidence: dict[str, Any]
