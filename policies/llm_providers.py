"""LLM provider abstraction for the Policy Agent RAG generation step.

The rest of the system (anomaly detection, bandit, compliance engine) makes
zero LLM calls by design -- see eval/cost_tracking.py for the full reasoning.
This module is the one place where provider choice matters, since natural
language policy Q&A genuinely needs a language model.

USAGE
-----
The active provider is controlled by a single env var:

    LLM_PROVIDER=anthropic   # default
    LLM_PROVIDER=openai
    LLM_PROVIDER=ollama

Each provider reads its own key/config from the environment (see .env.example).
The RAG logic in rag.py calls get_llm_response() and never needs to know
which provider is active -- it just gets back (answer, token_usage).

ADDING A NEW PROVIDER
---------------------
1. Implement a function matching the signature:
       _call_<name>(system: str, user: str, max_tokens: int) -> dict
   Return {"answer": str, "token_usage": dict | None, "status": "ok"}.
2. Add it to _PROVIDERS below.
3. Document the required env vars in .env.example.

No other file needs to change.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Anthropic (default)
# Required env: ANTHROPIC_API_KEY
# Optional env: ANTHROPIC_MODEL (default: claude-sonnet-4-6)
# ---------------------------------------------------------------------------

def _call_anthropic(system: str, user: str, max_tokens: int) -> dict:
    import anthropic  # already a project dependency

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except TypeError as e:
        # anthropic resolves the key lazily -- missing key surfaces here
        return {"answer": None, "token_usage": None, "status": f"no_api_key_configured: {e}"}
    except anthropic.AuthenticationError as e:
        return {"answer": None, "token_usage": None, "status": f"authentication_failed: {e}"}
    except anthropic.APIError as e:
        return {"answer": None, "token_usage": None, "status": f"api_error: {e}"}

    answer = next((b.text for b in response.content if b.type == "text"), "")
    return {
        "answer": answer,
        "token_usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# OpenAI  (example -- swap in gpt-4o or any OpenAI-hosted model)
# Required env: OPENAI_API_KEY
# Optional env: OPENAI_MODEL (default: gpt-4o)
# Install:  uv add openai
# ---------------------------------------------------------------------------

def _call_openai(system: str, user: str, max_tokens: int) -> dict:
    try:
        from openai import OpenAI, APIError, AuthenticationError
    except ImportError:
        return {
            "answer": None,
            "token_usage": None,
            "status": "openai_not_installed: run `uv add openai`",
        }

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = OpenAI()  # reads OPENAI_API_KEY automatically

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except AuthenticationError as e:
        return {"answer": None, "token_usage": None, "status": f"authentication_failed: {e}"}
    except APIError as e:
        return {"answer": None, "token_usage": None, "status": f"api_error: {e}"}

    answer = response.choices[0].message.content or ""
    usage = response.usage
    return {
        "answer": answer,
        "token_usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        },
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Ollama  (local inference, no API key required)
# Required env: OLLAMA_MODEL (default: llama3)
# Optional env: OLLAMA_BASE_URL (default: http://localhost:11434/v1)
# Ollama exposes an OpenAI-compatible /v1 endpoint -- no extra SDK needed.
# Install:  uv add openai  (reuses the OpenAI client pointed at localhost)
# ---------------------------------------------------------------------------

def _call_ollama(system: str, user: str, max_tokens: int) -> dict:
    try:
        from openai import OpenAI, APIError
    except ImportError:
        return {
            "answer": None,
            "token_usage": None,
            "status": "openai_not_installed: run `uv add openai` (used as Ollama client)",
        }

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OLLAMA_MODEL", "llama3")

    # Ollama's OpenAI-compatible endpoint requires a non-empty api_key value
    # but doesn't validate it -- "ollama" is the conventional placeholder.
    client = OpenAI(base_url=base_url, api_key="ollama")

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except APIError as e:
        return {"answer": None, "token_usage": None, "status": f"api_error: {e}"}
    except Exception as e:
        # Ollama not running locally is a connection error, not an API error
        return {"answer": None, "token_usage": None, "status": f"connection_error: {e}"}

    answer = response.choices[0].message.content or ""
    usage = response.usage
    token_usage = (
        {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens}
        if usage
        else None
    )
    return {"answer": answer, "token_usage": token_usage, "status": "ok"}


# ---------------------------------------------------------------------------
# Provider registry -- add new providers here only
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, callable] = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "ollama": _call_ollama,
}


def get_llm_response(system: str, user: str, max_tokens: int = 512) -> dict:
    """Single entry point for all LLM calls in this project.

    Reads LLM_PROVIDER from the environment (default: anthropic) and
    dispatches to the matching provider function. Returns a dict with
    keys: answer (str | None), token_usage (dict | None), status (str).
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()

    if provider not in _PROVIDERS:
        return {
            "answer": None,
            "token_usage": None,
            "status": (
                f"unknown_provider: '{provider}'. "
                f"Valid options: {list(_PROVIDERS.keys())}"
            ),
        }

    return _PROVIDERS[provider](system, user, max_tokens)
