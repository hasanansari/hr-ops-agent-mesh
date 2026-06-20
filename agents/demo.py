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
    for example in EXAMPLES:
        result = graph.invoke(example)
        print(f"\n=== request_id={result['request_id']} trigger={result['trigger_type']} ===")
        print(f"input:  {example.raw_input!r}")
        print(f"routed: {result['route']}")
        for entry in result["trace"]:
            print(f"  [{entry.agent}] in={entry.input} -> out={entry.output}")


if __name__ == "__main__":
    run_demo()
