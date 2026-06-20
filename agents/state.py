from __future__ import annotations

import operator
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field


class TriggerType(str, Enum):
    REACTIVE_QUERY = "reactive_query"
    SCHEDULED_SCAN = "scheduled_scan"
    SYSTEM_ALERT = "system_alert"


class AgentName(str, Enum):
    SUPERVISOR = "supervisor"
    POLICY = "policy_agent"
    ACTION = "action_agent"
    ANOMALY_DETECTION = "anomaly_detection_agent"
    COMPLIANCE = "compliance_agent"


class TraceEntry(BaseModel):
    agent: AgentName
    input: dict[str, Any]
    output: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HROpsState(BaseModel):
    """Shared graph state. Nodes only ever read this in and return partial
    updates to it -- there is no other channel for cross-agent communication.
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trigger_type: TriggerType
    raw_input: str

    route: AgentName | None = None

    policy_result: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None
    anomaly_result: dict[str, Any] | None = None
    compliance_result: dict[str, Any] | None = None

    # operator.add on a list field tells LangGraph to concatenate each
    # node's returned trace entries onto the existing list instead of
    # replacing it -- the default merge strategy for any field is overwrite.
    trace: Annotated[list[TraceEntry], operator.add] = Field(default_factory=list)
