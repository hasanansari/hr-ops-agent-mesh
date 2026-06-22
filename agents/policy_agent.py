from agents.state import AgentName, HROpsState, TraceEntry
from policies.rag import answer_policy_question


def policy_agent_node(state: HROpsState) -> dict:
    """Real RAG implementation: retrieves grounded chunks from the HR
    policy handbook (TF-IDF, see policies/retrieval.py) and asks Claude
    Sonnet 4.6 to answer using only those chunks. This is the one node in
    the entire graph that makes a real LLM call -- see policies/rag.py's
    docstring for why every other agent deliberately doesn't need one.

    Gracefully degrades if no ANTHROPIC_API_KEY is configured: retrieval
    still runs and is reported (it needs no API key at all), only the
    final generated answer is unavailable.
    """
    node_input = {"raw_input": state.raw_input}
    rag_result = answer_policy_question(state.raw_input)

    result = {
        "answer": rag_result["answer"],
        "citations": rag_result["citations"],
        "status": rag_result["status"],
    }
    trace_entry = TraceEntry(
        agent=AgentName.POLICY,
        input=node_input,
        output=result,
        token_usage=rag_result["token_usage"],
    )
    return {"policy_result": result, "trace": [trace_entry]}
