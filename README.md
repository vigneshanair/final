# ConflictGuard AI

A multi-agent framework for detecting and resolving conflicting evidence in
Retrieval-Augmented Generation (RAG) systems. Master's capstone project.

Standard RAG pipelines retrieve relevant passages and generate an answer
without checking whether those passages agree. In real document
collections, one policy may supersede another, two sources may describe
different populations, or two equally credible sources may directly
disagree — and a conventional RAG system will silently pick one answer, or
worse, blend incompatible claims. ConflictGuard AI instead extracts claims
from retrieved passages, detects and classifies conflicts between them,
evaluates the reliability of each source, chooses a conflict-specific
resolution strategy, and verifies that the final answer is actually
supported by the evidence before it is returned.

## Architecture

A coordinator runs the question through six specialized agents, each
returning structured, inspectable output rather than one unrestricted
prompt:

1. **Retrieval** — semantic search over the document collection (ChromaDB +
   `all-MiniLM-L6-v2` sentence embeddings).
2. **Claim Extraction** — converts each retrieved passage into discrete,
   comparable factual claims.
3. **Conflict Detection** — classifies the claims as agreeing, or as one of
   six conflict types.
4. **Source Evaluation** — scores each source's relevance, recency,
   authority, and applicability to the question.
5. **Resolution** — picks a strategy: prefer the newer or more authoritative
   source, explain conditional differences, present both positions, ask for
   clarification, or abstain.
6. **Verification** — checks that the final answer is actually supported by
   the selected evidence before it is returned.

Conflict categories: `direct_factual_conflict`, `temporal_conflict`,
`conditional_conflict`, `source_reliability_conflict`, `ambiguous_question`,
`insufficient_evidence`, `no_conflict`.

Two baselines are implemented for comparison, mirroring the project's
evaluation plan:

- **Standard RAG** (`app/rag/assistant.py`) — retrieves one passage and
  answers directly, with optional prompt-injection detection and PII
  redaction guardrails.
- **Single-agent conflict-aware RAG** (`app/rag/single_agent_conflict_rag.py`)
  — one LLM call handles conflict detection, classification, and resolution
  together, instead of the specialized multi-agent pipeline.

## Repository layout

```
app/
  agents/       Claim extraction, conflict detection, source evaluation,
                resolution, and verification agents, plus the coordinator
  rag/          Document loader, embeddings, retriever, and the two baselines
  security/     Prompt-injection detection and PII redaction guardrails
  evaluation/   Runners that score each configuration against benchmark sets
data/
  documents/    Source documents (policies etc.) with built-in conflicts
  test_cases/   Benchmark question sets, including a held-out "unseen" split
  benchmarks/   Additional conflict-pair benchmark data
results/        Metrics and per-case results produced by the evaluation runners
dashboard.py    Interactive Streamlit dashboard (live queries + evaluation results)
test_*.py       Manual/example scripts exercising individual components
```

## Setup

Requires Python and a locally running [Ollama](https://ollama.com) server
with `qwen2.5:3b` pulled (the model the agents call for extraction,
detection, resolution, and verification).

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

ollama serve                   # in a separate terminal
ollama pull qwen2.5:3b
```

## Running the dashboard

```bash
streamlit run dashboard.py
```

Opens an interactive UI with:

- **Live Query** — ask a question, pick a configuration (standard RAG,
  single-agent baseline, or ConflictGuard multi-agent), and see retrieved
  evidence, extracted claims, conflict type, source scores, resolution
  strategy, verification status, final answer, and citations.
- **Evaluation Results** — visualizes the metrics already saved in
  `results/`, comparing the single-agent and multi-agent configurations on
  the held-out benchmark.
- **About** — a short explanation of the agent pipeline.

## Running the evaluation

Each runner in `app/evaluation/` scores one configuration against a
benchmark set and writes results/metrics JSON and CSV to `results/`:

```bash
python -m app.evaluation.conflictguard_unseen_runner   # multi-agent, unseen set
python -m app.evaluation.single_agent_unseen_runner     # single-agent baseline
python -m app.evaluation.heldout_runner                 # security guardrail eval
```

On the 42-case held-out benchmark, the multi-agent pipeline reaches ~98%
conflict-type accuracy versus ~52% for the single-agent baseline — see the
dashboard's Evaluation Results tab for the full breakdown.

## Manual test scripts

The `test_*.py` scripts at the repo root exercise individual agents and
scenarios directly (e.g. `python test_coordinator.py`,
`python test_temporal_regression.py`) and are useful for spot-checking a
single component without running the full evaluation suite.
