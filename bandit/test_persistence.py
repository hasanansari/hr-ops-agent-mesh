"""Proof that the persisted policy survives a real process restart -- not
an in-memory claim. Run as two genuinely separate invocations:

    uv run python -m bandit.test_persistence train
    uv run python -m bandit.test_persistence verify

`train` creates a bandit, updates it, saves it to disk, and writes down
what it expects a fresh load to produce. That process then exits completely
-- nothing is held in memory afterward. `verify` is a brand new Python
process that has never seen the trained bandit; it only loads the saved
file and checks its behaviour matches what `train` wrote down.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bandit.policy import ACTIONS, LinearEpsilonGreedyBandit, context_vector

TEST_POLICY_PATH = Path(__file__).parent / "_persistence_test_policy.json"
TEST_EXPECTED_PATH = Path(__file__).parent / "_persistence_test_expected.json"

TEST_CONTEXT = context_vector("payroll_outlier", 0.9)


def train() -> None:
    bandit = LinearEpsilonGreedyBandit(epsilon=0.0, learning_rate=0.2, seed=42)
    for _ in range(30):
        bandit.update(TEST_CONTEXT, "escalate-to-HR", 1.0)
        for other in ACTIONS:
            if other != "escalate-to-HR":
                bandit.update(TEST_CONTEXT, other, -1.0)

    bandit.save(TEST_POLICY_PATH)

    expected = {
        "chosen_action": bandit.select_action(TEST_CONTEXT, explore=False),
        "scores": {a: bandit.predict(TEST_CONTEXT, a) for a in ACTIONS},
        "update_count": bandit.update_count,
    }
    with open(TEST_EXPECTED_PATH, "w") as f:
        json.dump(expected, f, indent=2)

    print(f"[train] saved policy to {TEST_POLICY_PATH}")
    print(f"[train] chosen action at test context: {expected['chosen_action']}")
    print(f"[train] scores: { {a: round(s, 4) for a, s in expected['scores'].items()} }")
    print("[train] process exiting now -- nothing is held in memory after this.")


def verify() -> None:
    if not TEST_POLICY_PATH.exists():
        print("[verify] FAIL -- no saved policy found, run `train` first")
        sys.exit(1)

    bandit = LinearEpsilonGreedyBandit.load(TEST_POLICY_PATH)
    with open(TEST_EXPECTED_PATH) as f:
        expected = json.load(f)

    actual_action = bandit.select_action(TEST_CONTEXT, explore=False)
    actual_scores = {a: bandit.predict(TEST_CONTEXT, a) for a in ACTIONS}

    action_match = actual_action == expected["chosen_action"]
    scores_match = all(abs(actual_scores[a] - expected["scores"][a]) < 1e-9 for a in ACTIONS)
    count_match = bandit.update_count == expected["update_count"]

    print(f"[verify] loaded policy from {TEST_POLICY_PATH} in a fresh process")
    print(f"[verify] chosen action: {actual_action} (expected {expected['chosen_action']}) -- match={action_match}")
    print(f"[verify] update_count: {bandit.update_count} (expected {expected['update_count']}) -- match={count_match}")
    print(f"[verify] scores match exactly: {scores_match}")

    if action_match and scores_match and count_match:
        print("[verify] PASS -- restart-safe: reloaded policy behaves identically to before save.")
    else:
        print("[verify] FAIL -- reloaded policy diverged from the saved state.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("train", "verify"):
        print("usage: python -m bandit.test_persistence [train|verify]")
        sys.exit(1)
    {"train": train, "verify": verify}[sys.argv[1]]()
