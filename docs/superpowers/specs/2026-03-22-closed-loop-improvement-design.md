# P1.1 — Closed-Loop Improvement Workflow Design Spec

**Date:** 2026-03-22
**Scope:** Trace persistence, inline feedback, trace dashboard, dataset builder, retrain pipeline, regression eval
**Status:** Approved

---

## Context

Pulsar AI has the building blocks for closed-loop improvement (agent traces via `data_gen.py`, feedback collector, observability tracer, pipeline executor, canary deployer) but they are disconnected. Traces live in memory only, feedback writes to JSONL, there's no way to go from "agent made a mistake" to "model is retrained and better" without manual work.

This spec connects the pieces into a product workflow: agents run → traces persist → users give feedback inline → traces get labeled → datasets are built → models retrain → regression eval gates deployment.

---

## Part 1: Trace Persistence Layer

### Database tables

Add to `src/pulsar_ai/storage/schema.py`:

**`traces` table:**
| Column | Type | Description |
|--------|------|-------------|
| trace_id | TEXT PK | UUID |
| agent_id | TEXT | Agent config name |
| model_name | TEXT | Model used |
| model_version | TEXT | Model version/path |
| user_query | TEXT | User input |
| response | TEXT | Agent final answer |
| trace_json | TEXT | Full ReAct trace (JSON) |
| status | TEXT | success / error |
| tokens_used | INTEGER | Total tokens |
| cost | REAL | Estimated cost USD |
| latency_ms | INTEGER | End-to-end latency |
| created_at | TEXT | ISO timestamp |

**`trace_feedback` table:**
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | UUID |
| trace_id | TEXT FK | References traces |
| feedback_type | TEXT | thumbs / rating / preference |
| rating | REAL | 0/1 for thumbs, 1-5 for rating |
| reason | TEXT | Optional: hallucination, wrong_tool, unsafe, slow |
| chosen | TEXT | For preference pairs |
| rejected | TEXT | For preference pairs |
| user_id | TEXT | Who gave feedback |
| created_at | TEXT | ISO timestamp |

### TraceStore

**`src/pulsar_ai/storage/trace_store.py`:**

```python
class TraceStore:
    def save_trace(self, trace_data: dict) -> str
    def get_trace(self, trace_id: str) -> dict | None
    def list_traces(self, filters: TraceFilter) -> list[dict]
    def add_feedback(self, trace_id: str, feedback_type: str, rating: float, reason: str = "", user_id: str = "") -> str
    def get_feedback(self, trace_id: str) -> list[dict]
    def export_as_sft(self, trace_ids: list[str]) -> list[dict]
    def export_as_dpo(self, good_ids: list[str], bad_ids: list[str]) -> list[dict]
    def auto_pair_dpo(self, trace_ids: list[str]) -> list[dict]
    def get_stats(self, days: int = 30) -> dict
```

**`TraceFilter` dataclass:**
- date_from, date_to: optional ISO strings
- model_name: optional
- status: optional (success/error)
- min_rating: optional float
- max_rating: optional float
- has_feedback: optional bool
- limit: int = 100
- offset: int = 0

**DPO auto-pairing logic (`auto_pair_dpo`):**
- Group traces by similar user_query (exact match or fuzzy via normalized text)
- Within each group: traces with rating >= threshold = chosen, rating < threshold = rejected
- Generate pairs: each (chosen, rejected) combination
- Deduplicate, filter empty responses

### Agent integration

Modify `src/pulsar_ai/agent/base.py`:
- After `run()` completes, auto-save trace via TraceStore
- Store trace_id on the agent instance: `self.last_trace_id`
- Lazy import TraceStore inside run() to avoid circular deps

---

## Part 2: Inline Feedback in Agent Chat

### UI changes

**`ui/src/components/FeedbackButtons.tsx`:**
- Small component rendered after each agent response in chat
- Two buttons: 👍 (thumbs up) / 👎 (thumbs down)
- On 👎 click: expand dropdown with reasons: hallucination, wrong_tool, unsafe, slow, other
- After click: button highlights, shows "Feedback saved"
- Props: `traceId: string`, `onFeedback: (traceId, type, rating, reason?) => void`

### Backend changes

**Modify `src/pulsar_ai/ui/routes/site_chat.py`:**
- Agent response now includes `trace_id` field
- Chat message format: `{ role: "assistant", content: "...", trace_id: "uuid-..." }`

**New route in `src/pulsar_ai/ui/routes/traces.py`:**
- `POST /api/v1/traces/{trace_id}/feedback` — body: `{ type, rating, reason?, user_id? }`

### Flow
1. User sends message in Agent Chat
2. Backend runs agent → trace auto-saved → trace_id returned with response
3. UI renders response + FeedbackButtons with trace_id
4. User clicks 👍 or 👎 → POST feedback → stored in trace_feedback table

---

## Part 3: Trace Dashboard

### New page: `ui/src/pages/Traces.tsx`

Added to sidebar navigation between "Agent Chat" and "Settings".

**List view (default):**
- Table: Time, Query (truncated 60 chars), Model, Status badge, Rating (👍/👎/—), Latency, Cost
- Filters bar: date range picker, model dropdown, status dropdown, rating filter
- Sort by any column
- Checkbox column for batch selection
- Pagination (50 per page)

**Detail view (click row → slide-out panel or modal):**
- Full user query
- Full agent response
- ReAct trace timeline:
  - Each step as a card: Thought → Action(tool, args) → Observation → ...
  - Collapsible tool call details
  - Color-coded: tool calls blue, observations gray, errors red
- Metrics: tokens, cost, latency
- Feedback history list
- Inline labeling buttons (same FeedbackButtons component)

**Batch operations (toolbar):**
- "Build SFT Dataset" button → dialog: name, confirm count → POST /api/v1/traces/build-dataset
- "Build DPO Dataset" button → auto-pairs by query match, shows preview of pairs count → POST
- "Export JSONL" → direct download
- After build: dialog "Created dataset v3 with 156 examples. [Train Now] [Close]"
- "Train Now" → creates experiment from dataset, navigates to experiment page

### Backend routes: `src/pulsar_ai/ui/routes/traces.py`

```python
router = APIRouter(tags=["traces"])

GET  /traces              — list with filters + pagination
GET  /traces/stats        — counts, avg rating, traces per day chart data
GET  /traces/{trace_id}   — full detail with feedback
POST /traces/{trace_id}/feedback — add feedback
POST /traces/build-dataset — { trace_ids, format: "sft"|"dpo", name } → creates versioned dataset
```

### Register in app.py
- Import and include traces router with `/api/v1` prefix
- Add "Traces" to sidebar nav in frontend

---

## Part 4: Closed-Loop Pipeline Steps

### New pipeline step types

Add to `src/pulsar_ai/pipeline/steps.py` (or new file `src/pulsar_ai/pipeline/closed_loop_steps.py`):

**`collect_traces` step:**
- Config: `{ days: 7, min_rating: 1, status: "success", model_name: "...", limit: 500 }`
- Queries TraceStore with filters
- Output: `{ trace_ids: [...], count: N }`

**`build_dataset` step:**
- Config: `{ format: "sft"|"dpo", quality_filter: { dedup: true, min_length: 10 }, source: "${collect.trace_ids}" }`
- Calls TraceStore.export_as_sft or auto_pair_dpo
- Writes JSONL to output path
- Registers in DatasetVersionStore
- Output: `{ path: "...", version: N, num_examples: N }`

**`regression_eval` step:**
- Config: `{ new_model: "${retrain.output_dir}", baseline_model: "./current_model", eval_dataset: "...", judge_criteria: [...] }`
- Runs LLM-as-Judge on both models with same eval dataset
- Computes: win_rate, avg_score_delta per criterion, regression_detected (bool)
- If regression_detected and `block_on_regression: true` → step fails, pipeline halts
- Output: `{ win_rate: 0.65, score_delta: +0.3, regression_detected: false, report: {...} }`

### New workflow node types

Add to personas.ts and FlowCanvas:
- **CollectTraces** — persona "Archivist" (category: observer, color: #6366f1 indigo)
- **BuildDataset** — persona "Curator" (category: engineer, color: #14b8a6 teal)
- **RegressionEval** — persona "Auditor" (category: scientist, color: #f59e0b amber)

### CLI shortcut

`pulsar retrain` command:
```bash
pulsar retrain --from-traces --days 7 --min-rating 1 --task dpo --model current_model
```
Equivalent to running collect_traces → build_dataset → train → regression_eval pipeline.

---

## Part 5: Tests

### `tests/test_trace_store.py`
- save_trace returns valid trace_id
- get_trace retrieves saved trace
- list_traces with filters (date, model, status, rating)
- add_feedback links to trace
- get_feedback returns feedback list
- export_as_sft generates correct format
- export_as_dpo generates pairs
- auto_pair_dpo groups by query and pairs good/bad
- get_stats returns counts and averages

### `tests/test_traces_api.py`
- GET /traces returns list with pagination
- GET /traces/{id} returns detail
- POST /traces/{id}/feedback saves feedback
- POST /traces/build-dataset creates dataset
- GET /traces/stats returns stats
- Filter by date range, model, rating works

### `tests/test_closed_loop_pipeline.py`
- collect_traces step queries TraceStore
- build_dataset step creates JSONL with correct format
- DPO auto-pairing produces valid pairs
- Quality filters remove duplicates and short traces
- regression_eval compares two model outputs
- Pipeline halts on regression

### `tests/test_feedback_inline.py`
- Agent chat response includes trace_id
- Feedback POST saves to trace_feedback table
- Feedback with reason saves reason field

### Edge case tests
- Empty traces collection → clear error message
- All traces same rating → can't build DPO → error with explanation
- Regression detected → pipeline stops with detailed report
- Trace with no feedback → excluded from rated filters, included in "unrated"
- Duplicate queries → deduplication in dataset builder

---

## File Inventory

### New files (12)
- `src/pulsar_ai/storage/trace_store.py`
- `src/pulsar_ai/pipeline/closed_loop_steps.py`
- `src/pulsar_ai/ui/routes/traces.py`
- `ui/src/pages/Traces.tsx`
- `ui/src/components/FeedbackButtons.tsx`
- `tests/test_trace_store.py`
- `tests/test_traces_api.py`
- `tests/test_closed_loop_pipeline.py`
- `tests/test_feedback_inline.py`

### Modified files (8)
- `src/pulsar_ai/storage/schema.py` — add traces + trace_feedback tables
- `src/pulsar_ai/agent/base.py` — auto-save trace after run()
- `src/pulsar_ai/ui/routes/site_chat.py` — return trace_id in response
- `src/pulsar_ai/ui/app.py` — register traces router
- `src/pulsar_ai/cli.py` — add `pulsar retrain` command
- `src/pulsar_ai/pipeline/executor.py` — register new step types
- `ui/src/components/flow/personas.ts` — add 3 new personas
- `ui/src/components/flow/FlowCanvas.tsx` — register 3 new node types
