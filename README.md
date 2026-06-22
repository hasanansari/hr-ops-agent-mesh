# Darwinbox Self-Healing HR Ops Platform

A multi-agent system that triages HR requests across three trigger types — reactive employee
queries, scheduled data scans, and system-generated alerts — routes them through specialized
agents that communicate only via shared graph state, and learns from human feedback over time.
Anomaly detection is statistical (z-score, not ML), action-selection is a contextual bandit that
warm-starts from episodic memory, and a YAML-defined compliance engine can hard-veto any decision —
including an explicit human approval — before it reaches execution.

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Running Each Component](#running-each-component)
- [Key Design Decisions](#key-design-decisions)
- [What I'd Change at Production Scale](#what-id-change-at-production-scale)
- [Project Structure](#project-structure)

---

## Architecture

Every node below is a plain function: `(state) -> partial_state_update`. No node ever calls another
node directly — the only channel between them is the shared `HROpsState` object, enforced
structurally (agent files never import each other; only `agents/graph.py` does). Every node is
wrapped in a timing decorator and appends to a shared trace list (**Section G**), which is
serialized to `traces/*.json` after every run.

```
                                  ┌──────────────────────┐
   reactive_query  ──────┐        │      SUPERVISOR       │   Section A
   scheduled_scan ───────┼───────▶│  triages by trigger    │
   system_alert   ──────┘        │  type + keyword match  │
                                  └───────────┬────────────┘
                                              │ writes `route` to shared state
                ┌─────────────────┬───────────┴───────────┬───────────────────┐
                ▼                 ▼                       ▼                   ▼
        ┌───────────────┐ ┌───────────────┐     ┌──────────────────┐ ┌──────────────────┐
        │ POLICY AGENT  │ │ ACTION AGENT  │     │ ANOMALY DETECTION│ │ COMPLIANCE AGENT │
        │ real RAG:     │ │ real tool     │     │ z-score (payroll)│ │ reactive path --  │
        │ TF-IDF + a    │ │ exec: 3 mock  │     │ + leave-pattern  │ │ stub (no NLP yet  │
        │ real Claude   │ │ HR tools,     │     │ + compliance     │ │ to parse a free-  │
        │ Sonnet 4.6    │ │ retry-wrapped │     │ rule checks      │ │ text alert into   │
        │ call          │ │               │     │                  │ │ structured input) │
        │ Section A     │ │ Section A     │     │ Section B        │ │ Section A         │
        └──────┬────────┘ └──────┬────────┘     └────────┬─────────┘ └─────────┬─────────┘
               │                 │                        ▼                     │
               │                 │              ┌───────────────────┐           │
               │                 │              │   BANDIT AGENT    │           │
               │                 │              │ epsilon-greedy    │           │
               │                 │              │ linear bandit,    │           │
               │                 │              │ warm-started from │           │
               │                 │              │ episodic memory   │           │
               │                 │              │ Section C + F     │           │
               │                 │              └─────────┬─────────┘           │
               │                 │                         ▼                     │
               │                 │              ┌───────────────────┐           │
               │                 │              │     HITL GATE     │           │
               │                 │              │ blocks, polling a │           │
               │                 │              │ shared SQLite     │           │
               │                 │              │ store the         │           │
               │                 │              │ Streamlit app     │           │
               │                 │              │ writes to         │           │
               │                 │              │ Section D         │           │
               │                 │              └─────────┬─────────┘           │
               │                 │                         ▼                     │
               │                 │              ┌───────────────────┐           │
               │                 │              │ COMPLIANCE VETO   │◀──────────┘
               │                 │              │ 15 YAML rules,    │
               │                 │              │ hard veto even    │
               │                 │              │ over an explicit  │
               │                 │              │ human approval    │
               │                 │              │ Section E         │
               │                 │              └────┬─────────┬────┘
               │                 │                    │         │
               │                 └────────────────────┘         │
               │                  (only if something is          │
               │                   actually actionable)          │
               ▼                                                  ▼          ▼
              END                                                END        END
```

**Where Section F (episodic memory) sits:** inside the Bandit Agent, not as its own node. Before
trusting its own linear weights, the bandit queries a Chroma collection of past resolved incidents
and biases toward whatever action worked well for similar ones — see
`memory/warm_start.py`.

**Where Section G (observability/eval) sits:** not in the graph at all — it wraps it. The
`_timed()` decorator in `agents/graph.py` stamps latency onto every node uniformly;
`agents/demo.py` writes the full trace to disk after every invocation; `eval/test_cases.py` and
`eval/cost_tracking.py` run independently against the same graph and dataset.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Package manager | [`uv`](https://docs.astral.sh/uv/) |
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) — `StateGraph`, conditional edges |
| State schema | Pydantic v2 |
| LLM | `anthropic` SDK, `claude-sonnet-4-6` (Policy Agent's RAG answer generation only) |
| Vector store | ChromaDB (episodic memory, Section F) — precomputed embeddings, no network-dependent embedding model |
| HITL UI | Streamlit |
| HITL persistence | SQLite (stdlib `sqlite3`) |
| Rules engine | PyYAML — structured `(field, operator, value)` conditions, no `eval()` |
| Numerics | NumPy (z-scores, bandit linear algebra, TF-IDF vectors) |
| API surface (scaffolded, unused by the CLI demo) | FastAPI, Uvicorn |

No ML framework, no scikit-learn, no sentence-transformers. Every "model" in this system — the
anomaly detector, the bandit, the rules engine, the memory retriever — is a formula or a linear
model simple enough to compute by hand. The one exception is Policy Agent's answer generation,
which is the one task in the whole pipeline that's actually LLM-shaped (see
[Key Design Decisions](#key-design-decisions)).

---

## Setup

```bash
# 1. Clone
git clone https://github.com/hasanansari/darwinbox-hr-ops-agent.git
cd darwinbox-hr-ops-agent

# 2. Install dependencies (creates .venv automatically, reads uv.lock)
uv sync

# 3. Configure environment (optional -- only needed for Policy Agent's live LLM call)
cp .env.example .env
# then edit .env and set:
#   ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the full system
uv run main.py
```

Without `ANTHROPIC_API_KEY` set, everything still runs end to end — Policy Agent's retrieval still
executes and reports its citations, it just can't generate a final answer, and reports a clear
`no_api_key_configured` status instead of crashing.

The synthetic dataset (`data/employees.csv`, 800 employees) is already committed and seeded
deterministically (`SEED=42`). Regenerate it with:

```bash
uv run python data/generate_employees.py
```

---

## Running Each Component

**The main graph** (all 3 trigger types, full pipeline, writes a trace per run to `traces/`):
```bash
uv run main.py
```

**The HITL review UI** (run in a separate terminal while `main.py`'s scheduled-scan example is
waiting — default window is 30s in the demo, 120s in production):
```bash
uv run streamlit run hitl/app.py
```

**The bandit training cycles** (demonstrates measurable learning across 2 feedback cycles, saves
the trained policy + RL diagnostics to disk):
```bash
uv run python -m bandit.train_cycles
```

**The persistence proof** (two genuinely separate process invocations — the second has no memory
of the first):
```bash
uv run python -m bandit.test_persistence train
uv run python -m bandit.test_persistence verify
```

**The episodic memory warm-start proof** (same untrained bandit, two occurrences, only memory
changes between them):
```bash
uv run python -m memory.demo_warm_start
```

**The evaluation harness** (15 tests: happy path, edge cases, adversarial inputs, RL-specific —
pass/fail with reasoning for each):
```bash
uv run python -m eval.test_cases
```

**The LLM cost analysis** (real Sonnet 4.6 pricing vs. a naive all-LLM baseline, using real call
counts from an actual scan):
```bash
uv run python -m eval.cost_tracking
```

---

## Key Design Decisions

**1. Epsilon-greedy linear bandit over LinUCB (Section C).** LinUCB explores by maintaining and
inverting a per-action covariance matrix to know how *uncertain* it is about each action — more
sample-efficient, but real machinery to defend. With 5 actions and a 5-feature context, uniform
random exploration covers the space almost as well, and the whole mechanism fits in one sentence:
score each action linearly, mostly take the best one, sometimes gamble, nudge the weights toward
the observed reward afterward. Chose the version I could fully explain over the version that's
marginally more sample-efficient.

**2. YAML rules over code or prompts (Section E).** Conditions are structured
`(field, operator, value)` triples, never an `eval()`'d expression string — a rules file editable
by a non-engineer can never become a code-injection surface. The tradeoff: no arbitrary logic, only
the 6 comparison operators the engine recognizes. For a compliance ruleset that's the right
tradeoff — auditability matters more than expressiveness here.

**3. Structured numeric embeddings over sentence-transformers, twice (Sections F and Policy RAG).**
Episodic memory embeds anomalies as a 6-dimensional hand-built feature vector; Policy Agent's
retrieval uses TF-IDF instead of a neural embedder. Same reasoning both times: no model download
(no guaranteed network access in this environment), and a similarity score that's fully
hand-verifiable instead of opaque. The real cost, demonstrated directly rather than just claimed:
TF-IDF retrieval missed a paraphrased query ("I just joined the company" vs. the document's
"probationary period") that a real embedding model likely would have caught — a known, accepted
limitation in exchange for explainability and zero infrastructure dependency.

**4. SQLite over a flat JSON file for HITL persistence (Section D).** Two separate OS processes —
the graph run and the Streamlit reviewer UI — read and write the same decisions concurrently. JSON
has no protection against two processes writing at once; SQLite gives transactions and row-level
locking for free while still being a single file on disk, no server to run.

**5. The compliance veto can override an explicit human approval (Section E).** The brief asked for
a hard veto overriding "the Supervisor or RL policy" — extended that to humans too, on purpose. A
hard compliance rule exists specifically to catch the case where a reviewer under-reacts to
something serious; if a human's approval could always have the final word, the veto would only be
advisory in practice. Verified directly: an approved `flag-for-audit` on a $7,000 payroll
discrepancy gets force-corrected to `escalate-to-HR` regardless of who signed off.

**6. The bandit augments the rule-based recommendation, it doesn't replace it (Section C).** Every
anomaly carries both `recommended_action` (Section B's tiered rule) and `bandit_action` (the
learned suggestion) side by side, and the gap between them is reported in the trace
(`agreement_with_rule_based`). Early on, before much training had accumulated, agreement was as low
as 19/178 — too unreliable to trust as the sole driver of a consequential HR action. Shipping the
learned policy as authoritative before it's earned that trust would be the wrong tradeoff; letting
both ride side by side until the bandit's track record is actually validated is the safer one.

**7. Sonnet 4.6, not Opus, for the one real LLM call (Policy Agent).** Grounded Q&A from a short
retrieved excerpt is a narrow, high-volume, well-scoped task — exactly Sonnet's profile — and it's
the same model Section G's cost analysis assumes for a platform processing millions of HR
transactions. Opus's extra capability would be paid for and mostly unused here.

---

## What I'd Change at Production Scale

**HITL would stop blocking a thread.** `hitl_gate_node` currently polls a SQLite store in a sleep
loop for up to the timeout window — fine for a CLI demo, but it holds a process hostage in a real
server. Production version: LangGraph's `interrupt()` + a checkpointer, fully suspending the run
and resuming it later via a webhook when a decision arrives, instead of occupying a thread the
whole time.

**The employee dataset would be a real database, not a CSV reloaded from disk on every call.**
`load_employees()` re-reads and re-parses the full CSV inside the Anomaly Detection, Bandit, and
Compliance Veto nodes on every single invocation. Fine at 800 rows; at the PDF's own stated scale
(3M+ employees) this needs a real database with indexing, not a flat file re-parsed per request.

**Episodic memory needs a retention policy.** The Chroma collection in `memory/store.py` only ever
grows — no pruning, no expiry, no archival of stale incidents. At real scale this needs a TTL or a
periodic compaction job, the same way a production logging pipeline doesn't keep every event
forever.

**Token cost tracking would use real counts, not an approximation.** `eval/cost_tracking.py`
estimates tokens at ~4 characters/token specifically to avoid live API calls. Production cost
tracking should pull from actual `usage.input_tokens` / `usage.output_tokens` on every real call
(which `policies/rag.py` already captures when a key is configured) rather than an estimate.

**Compliance overrides need their own audit trail.** Section D persists every human decision to
SQLite; Section E's overrides currently only live in graph state for the duration of one run --
there's no permanent record of "this anomaly's action was overridden from X to Y, for these
reasons" the way there is for human decisions. That's the natural next addition for Section G's
eventual evaluation harness.

**The RL feedback loop needs real, sustained human data, not primarily simulated data.** The real
HITL store today is almost entirely timeout fallbacks — there hasn't been enough actual human
review yet to train on. `bandit/simulate_human.py`'s synthetic feedback is doing the heavy lifting
for the current demonstration, which the brief explicitly sanctions given the sparsity, but a real
deployment needs a sustained flywheel of genuine approve/reject/modify decisions before the learned
policy should be trusted to influence real actions.

**Policy Agent's identity resolution is a placeholder.** There's no auth/session system, so reactive
self-service tool calls (`agents/action_agent.py`) resolve to a fixed demo employee ID rather than
the actual requester. Production needs this wired to whatever identity system fronts the platform.

---

## Project Structure

```
agents/        Supervisor, all 7 graph nodes, shared Pydantic state schema
data/          Synthetic employee dataset + generator (Section B)
bandit/        Contextual bandit, reward function, training cycles (Section C)
hitl/          SQLite store + Streamlit review UI (Section D)
compliance/    YAML rules + the eval()-free rules engine (Section E)
memory/        Episodic memory: embeddings, Chroma store, warm-start blending (Section F)
eval/          15-test harness + LLM cost analysis (Section G)
policies/      Mock HR policy doc, chunking, TF-IDF retrieval, real RAG generation
tools/         Mock HR self-service tool schemas + retry-wrapped implementations
docs/          (gitignored) in-depth notes per section, not part of the submission
traces/        Generated per-run JSON traces (gitignored)
```
