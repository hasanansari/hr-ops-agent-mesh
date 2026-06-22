from enum import Enum

from agents.anomaly_models import RecommendedAction


class HITLDecisionType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


# The 5 recommended actions aren't free text, they're 5 fixed category
# labels -- so plain string edit distance between them (e.g. Levenshtein on
# "escalate-to-manager" vs "escalate-to-HR") would just measure spelling
# overlap, not how big a disagreement the human actually made. Instead we
# rank the 5 actions on one dimension: how much autonomous/escalation power
# each one invokes, from least (0) to most (4). "Edit distance" is then the
# gap between those ranks -- a human picking a *nearby* rank made a mild
# severity correction; a *far* rank means the system badly misjudged how
# serious the situation was.
ACTION_RANK = {
    RecommendedAction.NO_ACTION: 0,
    RecommendedAction.FLAG_FOR_AUDIT: 1,
    RecommendedAction.ESCALATE_TO_MANAGER: 2,
    RecommendedAction.ESCALATE_TO_HR: 3,
    RecommendedAction.AUTO_CORRECT: 4,
}

# Timeout fallback default. "no-action" risks silently dropping a real
# problem just because nobody reviewed it in time; "flag-for-audit" keeps it
# visible for eventual human attention without taking any consequential
# automated step -- fail visible, not fail silent.
DEFAULT_TIMEOUT_ACTION = RecommendedAction.FLAG_FOR_AUDIT


def edit_distance(proposed: RecommendedAction, final: RecommendedAction) -> int:
    return abs(ACTION_RANK[final] - ACTION_RANK[proposed])
