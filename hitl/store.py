"""SQLite-backed persistence for HITL review decisions.

SQLite over a flat JSON file because two separate OS processes touch this
store concurrently: the graph run (writing pending items, polling for
resolution) and the Streamlit app (reading the queue, writing decisions).
A JSON file has no protection against two processes writing at once; SQLite
gives transactions and row-level locking essentially for free, while still
being "just a file on disk" -- no server process to run. It also means
querying decision history for the RL reward pipeline later (Section C) is
a SQL query instead of a hand-rolled file scan.
"""

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS hitl_decisions (
    anomaly_id TEXT PRIMARY KEY,
    request_id TEXT,
    employee_id TEXT NOT NULL,
    anomaly_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    proposed_action TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    status TEXT NOT NULL,              -- 'pending' | 'decided' | 'timeout_fallback'
    human_decision TEXT,               -- 'approve' | 'reject' | 'modify' | NULL
    final_action TEXT,
    rejection_reason TEXT,
    edit_distance INTEGER,
    is_timeout_fallback INTEGER NOT NULL DEFAULT 0,
    reviewer TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
)
"""


def _db_path() -> Path:
    # overridable so tests/demos can point at a throwaway file instead of
    # the real running queue
    override = os.environ.get("HITL_DB_PATH")
    return Path(override) if override else Path(__file__).parent / "decisions.sqlite"


def get_timeout_seconds() -> float:
    return float(os.environ.get("HITL_TIMEOUT_SECONDS", "120"))


def get_poll_interval_seconds() -> float:
    return float(os.environ.get("HITL_POLL_INTERVAL_SECONDS", "2"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(SCHEMA)
    return conn


def create_pending(anomaly: dict, request_id: str) -> None:
    """anomaly is an Anomaly.model_dump(). No-op if this anomaly_id already
    has a row -- safe to call repeatedly for the same scan result."""
    conn = _connect()
    with conn:
        conn.execute(
            """INSERT OR IGNORE INTO hitl_decisions
               (anomaly_id, request_id, employee_id, anomaly_type, confidence,
                proposed_action, evidence_json, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (
                anomaly["anomaly_id"],
                request_id,
                anomaly["employee_id"],
                anomaly["anomaly_type"],
                anomaly["confidence"],
                anomaly["recommended_action"],
                json.dumps(anomaly["evidence"]),
                datetime.now(UTC).isoformat(),
            ),
        )
    conn.close()


def get_status(anomaly_id: str) -> str | None:
    conn = _connect()
    row = conn.execute("SELECT status FROM hitl_decisions WHERE anomaly_id = ?", (anomaly_id,)).fetchone()
    conn.close()
    return row["status"] if row else None


def get_decision(anomaly_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM hitl_decisions WHERE anomaly_id = ?", (anomaly_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_pending() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM hitl_decisions WHERE status = 'pending' ORDER BY created_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_all() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM hitl_decisions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def submit_decision(
    anomaly_id: str,
    decision: str,
    final_action: str,
    rejection_reason: str | None = None,
    edit_distance_value: int | None = None,
    reviewer: str | None = None,
) -> None:
    # the "AND status = 'pending'" guard is what prevents a race between a
    # human clicking submit and the node's timeout sweep firing at the same
    # instant -- whichever UPDATE commits first wins, the loser matches zero
    # rows and silently no-ops instead of corrupting a decided record.
    conn = _connect()
    with conn:
        conn.execute(
            """UPDATE hitl_decisions
               SET status = 'decided', human_decision = ?, final_action = ?,
                   rejection_reason = ?, edit_distance = ?, reviewer = ?,
                   decided_at = ?
               WHERE anomaly_id = ? AND status = 'pending'""",
            (
                decision,
                final_action,
                rejection_reason,
                edit_distance_value,
                reviewer,
                datetime.now(UTC).isoformat(),
                anomaly_id,
            ),
        )
    conn.close()


def mark_timeout_fallback(anomaly_id: str, fallback_action: str) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """UPDATE hitl_decisions
               SET status = 'timeout_fallback', final_action = ?,
                   is_timeout_fallback = 1, decided_at = ?
               WHERE anomaly_id = ? AND status = 'pending'""",
            (fallback_action, datetime.now(UTC).isoformat(), anomaly_id),
        )
    conn.close()
