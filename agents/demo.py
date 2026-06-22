import os

from agents.graph import graph
from agents.state import HROpsState, TriggerType

EXAMPLES = [
    HROpsState(
        trigger_type=TriggerType.REACTIVE_QUERY,
        raw_input="What is our maternity leave policy?",
    ),
    HROpsState(
        trigger_type=TriggerType.REACTIVE_QUERY,
        raw_input="Apply for 3 days leave starting June 15",
    ),
    HROpsState(
        trigger_type=TriggerType.SCHEDULED_SCAN,
        raw_input="nightly cycle: scan attendance and payroll for anomalies",
    ),
    HROpsState(
        trigger_type=TriggerType.SYSTEM_ALERT,
        raw_input="payroll engine alert: overtime cap breach detected for EMP-4821",
    ),
]


def run_demo() -> None:
    # Production default is a 2-minute review window (HITL_TIMEOUT_SECONDS
    # in hitl/store.py); too long to sit through for a demo run, so it's
    # shortened here unless the caller already set their own. Run
    # `uv run streamlit run hitl/app.py` in another terminal during this
    # window to approve/reject/modify an item live and watch the routing
    # change.
    os.environ.setdefault("HITL_TIMEOUT_SECONDS", "30")
    os.environ.setdefault("HITL_POLL_INTERVAL_SECONDS", "2")

    for example in EXAMPLES:
        result = graph.invoke(example)
        print(
            f"\n=== request_id={result['request_id']} trigger={result['trigger_type']} ==="
        )
        print(f"input:  {example.raw_input!r}")
        print(f"routed: {result['route']}")
        for entry in result["trace"]:
            print(f"  [{entry.agent}] in={entry.input} -> out={entry.output}")

        anomaly_result = result.get("anomaly_result")
        if anomaly_result:
            print("  sample high-confidence anomalies (queued for HITL review):")
            for anomaly in anomaly_result["high_confidence_anomalies"][:3]:
                print(f"    {anomaly}")

        bandit_result = result.get("bandit_result")
        if bandit_result is not None:
            print(
                f"  bandit_result: agrees with rule-based recommendation on "
                f"{bandit_result['agree_count']}/{bandit_result['total']} anomalies"
            )

        hitl_result = result.get("hitl_result")
        if hitl_result is not None:
            print(
                f"  hitl_result: {len(hitl_result['reviewed'])} reviewed, "
                f"{len(hitl_result['approved_for_action'])} approved for Action Agent"
            )
            for decision in hitl_result["reviewed"][:5]:
                print(
                    f"    {decision['employee_id']} {decision['anomaly_type']}: "
                    f"proposed={decision['proposed_action']} final={decision['final_action']} "
                    f"human_decision={decision['human_decision']} "
                    f"timeout_fallback={bool(decision['is_timeout_fallback'])}"
                )


if __name__ == "__main__":
    run_demo()
