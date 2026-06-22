"""Episodic memory: a Chroma collection of past resolved incidents (context,
action taken, outcome, reward), keyed by the structured embedding from
embeddings.py. Embeddings are supplied explicitly on every add/query call,
so Chroma's own automatic embedding function (which would otherwise try to
download a sentence-transformer model) is never invoked -- see
embeddings.py for why that matters in this environment.

Persists to ./chroma_db at the project root -- already a name this repo's
.gitignore anticipated, so generated vector-store files never get committed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from memory.embeddings import embed_incident

DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
DEFAULT_COLLECTION = "episodic_memory"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(DB_PATH))
    return _client


def get_collection(name: str = DEFAULT_COLLECTION):
    return _get_client().get_or_create_collection(name=name, embedding_function=None)


def reset_collection(name: str) -> None:
    """Deletes a collection if it exists -- used by demo scripts that need
    a clean, reproducible starting state, not by normal read/write paths."""
    try:
        _get_client().delete_collection(name)
    except Exception:
        pass


def add_incident(
    anomaly_id: str,
    anomaly_type: str,
    confidence: float,
    evidence: dict,
    employee: dict | None,
    action_taken: str,
    is_true_positive: bool,
    is_timeout_fallback: bool,
    human_decision: str | None,
    reward: float,
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    collection = get_collection(collection_name)
    embedding = embed_incident(anomaly_type, confidence, evidence, employee)
    employee_id = (employee or {}).get("employee_id", "unknown")

    document = (
        f"{anomaly_type} for {employee_id}, confidence={confidence:.2f}, "
        f"action_taken={action_taken}, reward={reward:+.2f}"
    )
    metadata: dict[str, Any] = {
        "anomaly_type": anomaly_type,
        "confidence": float(confidence),
        "employee_id": employee_id,
        "action_taken": action_taken,
        "is_true_positive": bool(is_true_positive),
        "is_timeout_fallback": bool(is_timeout_fallback),
        "human_decision": human_decision or "",
        "reward": float(reward),
        "evidence_json": json.dumps(evidence),
    }
    # "upsert" semantics -- re-resolving the same anomaly_id (e.g. a re-run
    # of the same demo) replaces its memory record instead of duplicating it
    collection.upsert(ids=[anomaly_id], embeddings=[embedding], documents=[document], metadatas=[metadata])


def query_similar(
    anomaly_type: str,
    confidence: float,
    evidence: dict,
    employee: dict | None,
    k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict]:
    collection = get_collection(collection_name)
    if collection.count() == 0:
        return []

    embedding = embed_incident(anomaly_type, confidence, evidence, employee)
    result = collection.query(query_embeddings=[embedding], n_results=min(k, collection.count()))

    neighbors = []
    ids = result["ids"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    for incident_id, metadata, distance in zip(ids, metadatas, distances):
        neighbors.append(
            {
                "anomaly_id": incident_id,
                "distance": distance,
                "action_taken": metadata["action_taken"],
                "reward": metadata["reward"],
                "anomaly_type": metadata["anomaly_type"],
                "is_true_positive": metadata["is_true_positive"],
            }
        )
    return neighbors
