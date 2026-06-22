"""Loads compliance/rules.yaml and evaluates a proposed/final action against
it. Deliberately does NOT use eval() or any expression-string evaluation --
every condition is a structured (field, operator, value) triple, so a rule
file can never become a code-execution risk no matter who edits it later.
That's the actual reason conditions are (field, operator, value) tuples
instead of e.g. a "python:" expression column.

A veto here means exactly what the brief asks for: even if the Supervisor's
rule-based recommendation, the bandit's learned suggestion, or a human
reviewer's explicit decision picked an action, this engine can still force
it up to a stricter one. That includes overriding an explicit human
rejection (final_action forced to no-action) -- a deliberate choice, not an
oversight: a hard compliance rule existing specifically to catch a real
violation isn't optional just because a reviewer didn't flag it as one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from agents.anomaly_models import RecommendedAction
from hitl.models import ACTION_RANK, RANK_TO_ACTION

RULES_PATH = Path(__file__).parent / "rules.yaml"


class RuleScope(BaseModel):
    anomaly_types: list[str] | None = None
    violation: str | None = None


class RuleCondition(BaseModel):
    field: str
    operator: str
    value: Any


class Rule(BaseModel):
    rule_id: str
    description: str
    scope: RuleScope = RuleScope()
    condition: RuleCondition | None = None
    forbidden_actions: list[str] = []
    min_action_rank: int | None = None


class ComplianceVerdict(BaseModel):
    veto: bool
    violated_rule_ids: list[str]
    original_action: str
    final_action: str  # the action to actually use -- unchanged if veto=False


def load_rules(path: Path = RULES_PATH) -> list[Rule]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [Rule(**r) for r in data["rules"]]


def build_context(anomaly_type: str, confidence: float, evidence: dict, employee: dict | None) -> dict:
    context: dict[str, Any] = {
        "confidence": confidence,
        "anomaly_type": anomaly_type,
        "evidence": evidence,
        "employee": employee,
    }
    # precomputed/derived fields -- kept here, in code, rather than as a
    # free-form expression in the YAML, so rules stay pure data.
    if anomaly_type == "payroll_outlier" and "salary" in evidence and "cohort_mean" in evidence:
        context["payroll_discrepancy"] = abs(evidence["salary"] - evidence["cohort_mean"])
    return context


def _resolve(context: dict, field: str) -> Any:
    value: Any = context
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _condition_met(condition: RuleCondition | None, context: dict) -> bool:
    if condition is None:
        return True
    value = _resolve(context, condition.field)
    if value is None:
        return False  # field not resolvable (e.g. no employee record) -> can't evaluate, don't trigger
    op, target = condition.operator, condition.value
    if op == "<":
        return value < target
    if op == "<=":
        return value <= target
    if op == ">":
        return value > target
    if op == ">=":
        return value >= target
    if op == "==":
        return value == target
    if op == "between":
        return target[0] <= value < target[1]
    raise ValueError(f"unknown operator in rules.yaml: {op!r}")


def _scope_matches(scope: RuleScope, anomaly_type: str, evidence: dict) -> bool:
    if scope.anomaly_types is not None and anomaly_type not in scope.anomaly_types:
        return False
    if scope.violation is not None and evidence.get("violation") != scope.violation:
        return False
    return True


def evaluate(
    anomaly_type: str,
    confidence: float,
    evidence: dict,
    final_action: str,
    employee: dict | None = None,
    rules: list[Rule] | None = None,
) -> ComplianceVerdict:
    rules = rules if rules is not None else load_rules()
    context = build_context(anomaly_type, confidence, evidence, employee)
    action_rank = ACTION_RANK[RecommendedAction(final_action)]

    violated: list[Rule] = []
    for rule in rules:
        if not _scope_matches(rule.scope, anomaly_type, evidence):
            continue
        if not _condition_met(rule.condition, context):
            continue
        breaks_rank = rule.min_action_rank is not None and action_rank < rule.min_action_rank
        breaks_forbidden = final_action in rule.forbidden_actions
        if breaks_rank or breaks_forbidden:
            violated.append(rule)

    if not violated:
        return ComplianceVerdict(
            veto=False, violated_rule_ids=[], original_action=final_action, final_action=final_action
        )

    # the replacement is the gentlest action that still satisfies every
    # broken rule: start at the strictest minimum rank any violated rule
    # demands, then walk upward skipping anything any violated rule
    # specifically forbids.
    required_rank = max((r.min_action_rank for r in violated if r.min_action_rank is not None), default=action_rank)
    all_forbidden = {a for r in violated for a in r.forbidden_actions}
    replacement = None
    for rank in range(required_rank, 5):
        candidate = RANK_TO_ACTION[rank].value
        if candidate not in all_forbidden:
            replacement = candidate
            break
    if replacement is None:
        replacement = RecommendedAction.AUTO_CORRECT.value  # ceiling fallback, not reachable with this rule set

    return ComplianceVerdict(
        veto=True,
        violated_rule_ids=[r.rule_id for r in violated],
        original_action=final_action,
        final_action=replacement,
    )
