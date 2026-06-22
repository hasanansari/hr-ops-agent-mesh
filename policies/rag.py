"""The actual RAG generation step: retrieve grounded chunks, then ask
Claude to answer using only those chunks, citing which section(s) it used.

This is the one place in the entire project that makes a real LLM call --
every other agent (anomaly detection, the bandit, the compliance engine) is
deliberately statistical/rule-based/linear specifically so it *doesn't*
need one (see eval/cost_tracking.py for the full reasoning). Natural
language policy Q&A is different: grounding a free-text answer in retrieved
prose is a task an LLM is actually the right tool for, not a corner that
was cut elsewhere.

Model choice: claude-sonnet-4-6, not Opus. This is a narrow, well-scoped,
high-volume task (answer from a short excerpt, cite the source) -- exactly
the profile Sonnet is built for, and the one this project's own cost
analysis assumes for a production HR Ops platform processing millions of
transactions. Opus's extra capability would mostly be wasted here.
"""

from __future__ import annotations

import os

import anthropic

from policies.retrieval import build_index

MODEL = "claude-sonnet-4-6"
TOP_K = 3
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are the HR policy assistant for an HR Ops platform. Answer the employee's question "
    "using ONLY the policy excerpts provided below -- do not use outside knowledge, and do not "
    "guess. If the excerpts don't cover the question, say so plainly rather than speculating. "
    "Always state which section number(s) your answer is grounded in."
)


def _build_user_message(question: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(f"--- {c['title']} ---\n{c['text']}" for c in chunks)
    return f"Policy excerpts:\n\n{excerpts}\n\nEmployee question: {question}"


def answer_policy_question(question: str) -> dict:
    """Returns a dict with the answer (or a clear status if no live API key
    is configured), the retrieved chunks (always populated -- retrieval
    doesn't need an API key and is worth showing even when generation
    can't run), and real token usage when a call actually succeeds.
    """
    index = build_index()
    retrieved = index.query(question, k=TOP_K)

    if not retrieved:
        return {
            "answer": "I couldn't find anything in the policy handbook related to that question.",
            "citations": [],
            "status": "no_relevant_chunks",
            "token_usage": None,
        }

    # anthropic.Anthropic() itself never raises for a missing key -- it
    # resolves credentials lazily. The actual "no key at all" error only
    # surfaces as a plain TypeError from inside messages.create(), the
    # first time it tries to build request headers. That's why this can't
    # be caught at construction time; it has to be caught around the call.
    client = anthropic.Anthropic()

    user_message = _build_user_message(question, retrieved)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except TypeError as e:
        return {
            "answer": None,
            "citations": [c["title"] for c in retrieved],
            "status": f"no_api_key_configured: {e}",
            "retrieved_chunks": retrieved,
            "token_usage": None,
        }
    except anthropic.AuthenticationError as e:
        return {
            "answer": None,
            "citations": [c["title"] for c in retrieved],
            "status": f"authentication_failed: {e}",
            "retrieved_chunks": retrieved,
            "token_usage": None,
        }
    except anthropic.APIError as e:
        return {
            "answer": None,
            "citations": [c["title"] for c in retrieved],
            "status": f"api_error: {e}",
            "retrieved_chunks": retrieved,
            "token_usage": None,
        }

    answer_text = next((b.text for b in response.content if b.type == "text"), "")
    return {
        "answer": answer_text,
        "citations": [c["title"] for c in retrieved],
        "status": "ok",
        "retrieved_chunks": retrieved,
        "token_usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
