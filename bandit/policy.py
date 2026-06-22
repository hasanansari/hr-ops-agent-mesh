"""Contextual bandit for action-selection on detected anomalies.

Algorithm: epsilon-greedy with a linear value estimate per action, not
LinUCB. Both are standard contextual-bandit choices; LinUCB picks its
exploration via an uncertainty bonus (it maintains and inverts a per-action
covariance matrix to know how *unsure* it is about each action, then
explores the ones it's least sure about) which is more sample-efficient,
but that machinery is real complexity -- a matrix inversion per action per
update -- that has to be justified by the payoff. With only 5 actions and a
5-dimensional context, uniform random exploration (epsilon) covers the
action space almost as well without the linear-algebra overhead, and -- the
deciding factor -- it's something you can describe in one sentence with no
hand-waving: "mostly pick the action this linear score function rates
highest; sometimes pick randomly so we keep learning; after seeing the
reward, nudge the score function a little toward what actually happened."
That matches the project's running theme: pick the simplest method that is
still honestly defensible, not the most sophisticated one available.

Context vector layout (see context_vector()): a bias term plus a one-hot
encoding of anomaly_type plus the detector's confidence score. Kept
deliberately small and uniform across all three anomaly types -- richer,
type-specific features (e.g. z-score magnitude, leave days over limit)
would need separate feature engineering per type, which is a reasonable
next step but not done here to keep the model explainable in one sitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACTIONS: list[str] = [
    "auto-correct",
    "escalate-to-manager",
    "escalate-to-HR",
    "flag-for-audit",
    "no-action",
]

FEATURE_NAMES: list[str] = [
    "bias",
    "is_payroll_outlier",
    "is_leave_abuse",
    "is_compliance_violation",
    "confidence",
]
N_FEATURES = len(FEATURE_NAMES)


def context_vector(anomaly_type: str, confidence: float) -> np.ndarray:
    return np.array(
        [
            1.0,
            1.0 if anomaly_type == "payroll_outlier" else 0.0,
            1.0 if anomaly_type == "leave_abuse" else 0.0,
            1.0 if anomaly_type == "compliance_violation" else 0.0,
            float(confidence),
        ]
    )


class LinearEpsilonGreedyBandit:
    def __init__(
        self,
        epsilon: float = 0.15,
        learning_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        # one linear weight vector per action, all starting at zero -- a
        # brand-new bandit has no opinion yet, every action scores 0.
        self.weights: dict[str, np.ndarray] = {a: np.zeros(N_FEATURES) for a in ACTIONS}
        self.update_count = 0

    def predict(self, context: np.ndarray, action: str) -> float:
        return float(np.dot(self.weights[action], context))

    def select_action(self, context: np.ndarray, explore: bool = True) -> str:
        if explore and self.rng.random() < self.epsilon:
            return str(self.rng.choice(ACTIONS))

        scores = {a: self.predict(context, a) for a in ACTIONS}
        best_score = max(scores.values())
        # at cold start every score is exactly 0.0 -- argmax alone would
        # always return the first action in ACTIONS, which would look like
        # a (fake) preference rather than the genuine "no opinion yet" it
        # actually is. Breaking ties randomly keeps cycle-1 behaviour honest.
        tied = [a for a, s in scores.items() if s == best_score]
        return str(self.rng.choice(tied))

    def update(self, context: np.ndarray, action: str, reward: float) -> None:
        # plain online linear regression via one SGD step: predict the
        # reward, see how wrong that prediction was, nudge the weight
        # vector a little in the direction that would have predicted this
        # reward better next time.
        predicted = self.predict(context, action)
        error = reward - predicted
        self.weights[action] = self.weights[action] + self.learning_rate * error * context
        self.update_count += 1

    def to_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "update_count": self.update_count,
            "feature_names": FEATURE_NAMES,
            "weights": {a: w.tolist() for a, w in self.weights.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path, seed: int | None = None) -> "LinearEpsilonGreedyBandit":
        with open(path) as f:
            data = json.load(f)
        bandit = cls(epsilon=data["epsilon"], learning_rate=data["learning_rate"], seed=seed)
        bandit.weights = {a: np.array(w) for a, w in data["weights"].items()}
        bandit.update_count = data["update_count"]
        return bandit

    @classmethod
    def load_or_new(cls, path: Path, **kwargs) -> "LinearEpsilonGreedyBandit":
        if Path(path).exists():
            return cls.load(path, seed=kwargs.get("seed"))
        return cls(**kwargs)
