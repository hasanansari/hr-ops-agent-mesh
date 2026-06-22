# Architecture Brief: Darwinbox Self-Healing HR Ops Platform

## What it does

A multi-agent system that triages HR requests across three trigger types: reactive employee
queries, scheduled data scans, and system-generated alerts. Anomaly detection is statistical
(z-score against a peer cohort, not ML), action-selection is a contextual bandit that warm-starts
from past resolved incidents, and a YAML-defined compliance engine can hard-veto any decision,
including an explicit human approval, before it reaches execution. The one place that calls a real
LLM is policy Q&A, retrieval-grounded over a mock HR handbook; everything else is a formula simple
enough to verify by hand.

## Architecture at a glance

```
            ┌──────────────┐
  request ─▶│  SUPERVISOR  │  triages by trigger type + keyword
            └──────┬───────┘
       ┌───────────┼────────────┬─────────────┐
       ▼            ▼            ▼             ▼
   POLICY        ACTION      ANOMALY      COMPLIANCE
  (real RAG)   (real tool   DETECTION      (reactive
               execution)  (z-score/leave/   stub)
                            overtime)
                                │
                                ▼
                             BANDIT  ── warm-started from
                                       episodic memory (Chroma)
                                │
                                ▼
                            HITL GATE  ── human review via
                                          Streamlit + SQLite
                                │
                                ▼
                       COMPLIANCE VETO  ── 15 YAML rules,
                                           hard veto, even over
                                           a human approval
                                │
                          ┌─────┴─────┐
                          ▼           ▼
                       ACTION        END
```

Every node communicates only through shared graph state (LangGraph `StateGraph`); no node ever
calls another directly, enforced structurally by which files import which. Every node's
input/output/latency is appended to a shared trace, written to disk per run.

## The decisions that matter most

- **Anomaly detection is z-score, not a model.** No training data exists for "is this a real HR
  incident," and a statistical threshold is explainable to a non-technical reviewer in one sentence.
- **The bandit augments the rule-based recommendation, it doesn't replace it.** Both ride alongside
  each other in the trace until the learned policy has enough validated track record to be trusted
  alone.
- **Every embedding in this system (episodic memory, policy retrieval) is a hand-built numeric
  vector or TF-IDF, never a neural embedder.** No model download, no opaque similarity score;
  every "is this similar" judgment is hand-verifiable.
- **Compliance rules are YAML, not code or prompts.** Conditions are structured
  `(field, operator, value)` triples, never an `eval()`'d string, so the ruleset can't become a
  code-injection surface no matter who edits it.

## The hardest trade-off

The compliance veto can override an **explicit human approval**, not just the rule-based or
learned-policy recommendation. A reviewer approving `flag-for-audit` on a $7,000 payroll
discrepancy still gets force-corrected to `escalate-to-HR`, regardless of who signed off. That's
deliberate, not a bug: a hard compliance rule exists specifically to catch the case where a human
under-reacts to something serious, and if a human's approval always had the final word, the veto
would only be advisory in practice. The cost of that choice is real, though: it means this system
can, in a real deployment, override a manager's considered judgment, which needs to be a visible,
explainable event to that manager, not a silent correction. That's the actual tension: safety
against under-enforcement, traded against a human reviewer's sense of authority over their own
decision.

## What I'd change at production scale

- **HITL stops blocking a thread.** Swap the polling loop for LangGraph's `interrupt()` +
  checkpointer, so a real server isn't holding a process for up to 2 minutes per scan.
- **The CSV becomes a real database.** Three different nodes re-parse the full employee dataset
  from disk on every call, fine at 800 rows, not at the platform's actual stated scale (3M+
  employees).
- **Episodic memory gets a retention policy.** The vector store only ever grows today; production
  needs pruning or a TTL, the same way a logging pipeline doesn't keep every event forever.
- **The RL feedback loop needs real, sustained human data.** Today's real HITL history is almost
  entirely timeout fallbacks; simulated feedback is doing the heavy lifting for the demonstration,
  which is fine for now but not a substitute for an actual flywheel of human decisions over time.
