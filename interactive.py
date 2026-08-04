from dotenv import load_dotenv

from agents.graph import graph
from agents.state import HROpsState, TriggerType


def main():
    load_dotenv()
    print("Type an HR query and press enter (empty line to quit).")
    while True:
        query = input("\nquery> ").strip()
        if not query:
            break

        result = graph.invoke(
            HROpsState(trigger_type=TriggerType.REACTIVE_QUERY, raw_input=query)
        )

        print(f"routed to: {result['route']}")
        for entry in result["trace"]:
            print(f"  [{entry.agent}] in={entry.input} -> out={entry.output}")


if __name__ == "__main__":
    main()
