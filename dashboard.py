"""ConflictGuard AI interactive dashboard.

Run with: streamlit run dashboard.py

Lets a user ask a question against the document collection and compare how
three configurations handle it: a standard RAG baseline, a single-agent
conflict-aware baseline, and the proposed ConflictGuard multi-agent
pipeline. Also surfaces the offline evaluation results produced by the
runners in app/evaluation/.
"""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from app.agents.claim_extractor import ClaimExtractionAgent
from app.agents.coordinator import ConflictGuardCoordinator
from app.rag.assistant import RAGAssistant
from app.rag.retriever import Retriever
from app.rag.single_agent_conflict_rag import SingleAgentConflictRAG

RESULTS_DIR = Path("results")
EVAL_CASES_PATH = Path("data/test_cases/conflictguard_eval.json")

STANDARD_RAG = "Standard RAG (baseline)"
SINGLE_AGENT = "Single-Agent Conflict-Aware RAG (baseline)"
MULTI_AGENT = "ConflictGuard Multi-Agent (proposed)"

METRIC_LABELS = {
    "conflict_type_accuracy": "Conflict-Type Accuracy",
    "macro_f1": "Macro F1",
    "resolution_strategy_accuracy": "Resolution Strategy Accuracy",
    "abstention_accuracy": "Abstention Accuracy",
    "overall_pipeline_accuracy": "Overall Pipeline Accuracy",
    "average_latency_seconds": "Avg Latency (s)",
}

st.set_page_config(page_title="ConflictGuard AI", layout="wide")


# ---------------------------------------------------------------------
# Cached, expensive resources -- built once per server process
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading document index...")
def get_retriever():
    return Retriever()


@st.cache_resource(show_spinner="Loading standard RAG baseline...")
def get_standard_rag(protected):
    return RAGAssistant(protected=protected)


@st.cache_resource(show_spinner="Loading claim extraction agent...")
def get_claim_agent():
    return ClaimExtractionAgent()


@st.cache_resource(show_spinner="Loading single-agent baseline...")
def get_single_agent():
    return SingleAgentConflictRAG()


@st.cache_resource(show_spinner="Loading ConflictGuard multi-agent pipeline...")
def get_coordinator():
    return ConflictGuardCoordinator()


@st.cache_data
def load_example_questions():
    if not EVAL_CASES_PATH.exists():
        return []
    cases = json.loads(EVAL_CASES_PATH.read_text())
    return [(case["question"], case["category"]) for case in cases]


@st.cache_data
def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------
# Shared retrieval + claim extraction (used by the single-agent and
# multi-agent live-query modes)
# ---------------------------------------------------------------------

def retrieve_evidence(question, n_results):
    retriever = get_retriever()
    claim_agent = get_claim_agent()

    results = retriever.search(question, n_results=n_results)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    evidence = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        source = metadata.get("source", f"document_{index + 1}")
        extracted = claim_agent.extract(document, source=source)

        evidence.append({
            "source": source,
            "text": document,
            "publication_year": extracted.get("publication_year"),
            "policy_version": extracted.get("policy_version"),
            "source_authority": extracted.get("source_authority"),
            "document_status": extracted.get("document_status"),
            "document_condition": extracted.get("document_condition"),
            "claims": extracted.get("claims", []),
        })

    return evidence


def run_standard_rag(question, protected):
    return get_standard_rag(protected).answer(question)


def run_single_agent(question, n_results):
    evidence = retrieve_evidence(question, n_results)
    result = get_single_agent().answer(question, evidence)
    result["evidence"] = evidence
    return result


def run_multi_agent(question, n_results):
    return get_coordinator().answer(question, n_results=n_results)


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------

def render_evidence(evidence):
    st.subheader("Retrieved Evidence")

    if not evidence:
        st.info("No relevant documents were retrieved.")
        return

    rows = [{
        "Source": item["source"],
        "Publication Year": item.get("publication_year") or "-",
        "Policy Version": item.get("policy_version") or "-",
        "Authority": item.get("source_authority") or "-",
        "Status": item.get("document_status") or "-",
        "Condition": item.get("document_condition") or "-",
        "Claims Extracted": len(item.get("claims", [])),
    } for item in evidence]

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    for item in evidence:
        with st.expander(f"View passage and claims: {item['source']}"):
            if item.get("text"):
                st.text(item["text"])

            claims = item.get("claims", [])
            if claims:
                st.caption("Extracted claims")
                st.dataframe(pd.DataFrame(claims), width="stretch", hide_index=True)
            else:
                st.caption("No claims were extracted from this passage.")


def render_conflict(conflict_detected, conflict_type, explanation):
    st.subheader("Conflict Detection")

    cols = st.columns([1, 2])
    with cols[0]:
        if conflict_detected:
            st.error(f"Conflict detected: {conflict_type}")
        else:
            st.success(f"No blocking conflict ({conflict_type})")
    with cols[1]:
        if explanation:
            st.write(explanation)


def render_source_evaluation(source_result):
    st.subheader("Source Evaluation")

    evaluations = source_result.get("source_evaluations", [])
    preferred = source_result.get("preferred_source")

    if not evaluations:
        st.info("No source-level evaluation was produced for this question.")
        return

    rows = [{
        "Source": entry.get("source"),
        "Preferred": "Yes" if entry.get("source") == preferred else "",
        "Relevance": entry.get("relevance"),
        "Year": entry.get("publication_year"),
        "Version": entry.get("policy_version"),
        "Supersedes": entry.get("superseded_source") or "-",
        "Applicability": entry.get("applicability"),
    } for entry in evaluations]

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if preferred:
        st.caption(f"Preferred source: **{preferred}** — {source_result.get('preference_reason', '')}")


def render_resolution(resolution):
    st.subheader("Resolution")

    st.write(f"**Strategy:** {resolution.get('resolution_strategy', '-')}")
    if resolution.get("selected_source"):
        st.write(f"**Selected source:** {resolution['selected_source']}")
    st.caption(resolution.get("decision_reason", ""))


def render_verification(verification):
    st.subheader("Verification")

    if verification.get("verified"):
        st.success(f"Verified — {verification.get('final_status')}")
    else:
        st.warning(f"Not verified — {verification.get('final_status')}")
    st.caption(verification.get("verification_reason", ""))


def render_final_answer(answer, abstained, citations):
    st.subheader("Final Answer")

    if abstained:
        st.warning(answer)
    else:
        st.success(answer)

    if citations:
        st.caption("Citations: " + ", ".join(citations))


def render_standard_rag_result(result):
    st.subheader("Retrieved Sources")
    if result.get("sources"):
        st.write(", ".join(result["sources"]))
    else:
        st.info("No sources retrieved.")

    if result.get("blocked"):
        st.error(result["answer"])
        st.caption("Blocked by the prompt-injection guardrail before reaching the model.")
        with st.expander("Prompt-risk details"):
            st.json(result.get("prompt_risk", {}))
        return

    render_final_answer(result.get("answer", ""), False, result.get("sources", []))

    if result.get("detected_sensitive_terms"):
        st.warning("Redacted sensitive terms: " + ", ".join(result["detected_sensitive_terms"]))


def render_single_agent_result(result):
    render_evidence(result.get("evidence", []))
    render_conflict(
        result.get("conflict_detected", False),
        result.get("conflict_type", "-"),
        result.get("explanation", ""),
    )

    st.subheader("Resolution")
    st.write(f"**Strategy:** {result.get('resolution_strategy', '-')}")
    if result.get("selected_source"):
        st.write(f"**Selected source:** {result['selected_source']}")

    citations = [result["selected_source"]] if result.get("selected_source") else []
    render_final_answer(result.get("answer", ""), result.get("abstain", False), citations)


def render_multi_agent_result(result):
    render_evidence(result.get("evidence", []))
    render_conflict(
        result.get("conflict_detected", False),
        result.get("conflict_type", "-"),
        result.get("conflict_explanation", ""),
    )
    render_source_evaluation(result.get("source_evaluation", {}))
    render_resolution(result.get("resolution", {}))
    render_verification(result.get("verification", {}))

    citations = (
        [result["selected_source"]]
        if result.get("selected_source")
        else [item["source"] for item in result.get("evidence", [])]
    )
    render_final_answer(result.get("final_answer", ""), result.get("abstained", False), citations)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

def render_live_query_tab():
    st.write(
        "Ask a question against the ConflictGuard document set and see how each "
        "configuration handles it."
    )

    mode = st.radio("Configuration", [STANDARD_RAG, SINGLE_AGENT, MULTI_AGENT], horizontal=True)

    examples = load_example_questions()
    example_labels = ["(type your own)"] + [f"{question}  [{category}]" for question, category in examples]
    choice = st.selectbox("Example questions (one per conflict category)", example_labels)

    default_question = choice.split("  [")[0] if choice != "(type your own)" else ""

    question = st.text_input(
        "Question",
        value=default_question,
        placeholder="e.g. How many days per week can employees work remotely?",
    )

    col1, col2 = st.columns(2)
    with col1:
        n_results = st.slider(
            "Passages to retrieve", 1, 5, 3,
            disabled=(mode == STANDARD_RAG),
            help="Standard RAG always retrieves a single top passage.",
        )
    with col2:
        protected = st.checkbox(
            "Enable security guardrails",
            value=True,
            disabled=(mode != STANDARD_RAG),
            help="Prompt-injection detection and PII redaction, standard RAG mode only.",
        )

    if st.button("Run", type="primary", disabled=not question.strip()):
        with st.spinner("Running..."):
            try:
                start = time.perf_counter()

                if mode == STANDARD_RAG:
                    result = run_standard_rag(question, protected)
                elif mode == SINGLE_AGENT:
                    result = run_single_agent(question, n_results)
                else:
                    result = run_multi_agent(question, n_results)

                elapsed = time.perf_counter() - start

            except Exception as error:
                st.error(
                    "The request failed. Make sure Ollama is running locally "
                    "(`ollama serve`) and that the `qwen2.5:3b` model is pulled "
                    "(`ollama pull qwen2.5:3b`)."
                )
                st.exception(error)
                return

            st.session_state["last_result"] = (mode, question, result, elapsed)

    if "last_result" in st.session_state:
        mode, answered_question, result, elapsed = st.session_state["last_result"]
        st.divider()
        st.caption(f"Question: “{answered_question}” — mode: {mode} — latency: {elapsed:.2f}s")

        if mode == STANDARD_RAG:
            render_standard_rag_result(result)
        elif mode == SINGLE_AGENT:
            render_single_agent_result(result)
        else:
            render_multi_agent_result(result)


def render_evaluation_tab():
    st.write(
        "Offline results from the runners in `app/evaluation/`, comparing the "
        "single-agent baseline against the ConflictGuard multi-agent pipeline "
        "on a held-out, unseen benchmark."
    )

    multi = load_json_if_exists(RESULTS_DIR / "conflictguard_unseen_metrics.json")
    single = load_json_if_exists(RESULTS_DIR / "single_agent_unseen_metrics.json")

    if not multi and not single:
        st.info("No evaluation results found in `results/`. Run the evaluation runners in `app/evaluation/` first.")
        return

    if multi and single:
        st.subheader("Single-Agent vs. Multi-Agent (unseen benchmark)")

        comparison_df = pd.DataFrame([
            {
                "Metric": label,
                "Single-Agent Baseline": single.get(key),
                "ConflictGuard Multi-Agent": multi.get(key),
            }
            for key, label in METRIC_LABELS.items()
        ]).set_index("Metric")

        st.dataframe(comparison_df, width="stretch")

        accuracy_df = comparison_df.drop(index="Avg Latency (s)", errors="ignore")
        st.bar_chart(accuracy_df)

        if "Avg Latency (s)" in comparison_df.index:
            st.caption("Average latency comparison (seconds per question)")
            st.bar_chart(comparison_df.loc[["Avg Latency (s)"]].T)

        st.caption(
            f"Single-agent: {single.get('total_cases')} cases · "
            f"Multi-agent: {multi.get('total_cases')} cases"
        )

    if multi and multi.get("per_category"):
        st.subheader("Per-Category Breakdown (ConflictGuard Multi-Agent, unseen)")
        st.dataframe(pd.DataFrame(multi["per_category"]).T, width="stretch")

    small_eval = load_json_if_exists(RESULTS_DIR / "conflictguard_eval_metrics.json")
    if small_eval:
        with st.expander("Hand-crafted development set (one case per category)"):
            st.json(small_eval)

    external = load_json_if_exists(RESULTS_DIR / "external_conflicts_metrics.json")
    if external:
        st.subheader("External Evaluation (rag_conflicts benchmark)")
        st.caption(
            "A held-out sample from Google Research's rag_conflicts dataset, run through "
            "app/evaluation/external_conflicts_runner.py. Labels are mapped onto "
            "ConflictGuard's taxonomy where a defensible mapping exists; see "
            "data/benchmarks/build_external_sample.py for details."
        )

        cols = st.columns(4)
        cols[0].metric("Conflict-Type Accuracy", f"{external['conflict_type_accuracy'] * 100:.1f}%")
        cols[1].metric("Macro F1", f"{external['macro_f1']:.3f}")
        cols[2].metric("Abstain Rate", f"{external['abstain_rate'] * 100:.1f}%")
        cols[3].metric("Avg Latency", f"{external['average_latency_seconds']:.1f}s")

        st.caption(
            f"{external['scored_cases']} scored cases, "
            f"{external['unscored_cases']} unscored (no taxonomy equivalent) out of "
            f"{external['total_cases']} total"
        )

        if external.get("per_category"):
            st.dataframe(pd.DataFrame(external["per_category"]).T, width="stretch")

        with st.expander("Unscored cases and full details"):
            st.caption(external.get("unscored_note", ""))
            st.json(external.get("unscored_predicted_distribution", {}))

    security = load_json_if_exists(RESULTS_DIR / "heldout_metrics.json")
    if security:
        with st.expander("Security guardrail evaluation (prompt-injection detection)"):
            st.dataframe(pd.DataFrame([security]), width="stretch", hide_index=True)

    with st.expander("Raw result files in results/"):
        for path in sorted(RESULTS_DIR.glob("*.csv")):
            st.write(f"`{path}`")
        for path in sorted(RESULTS_DIR.glob("*.json")):
            st.write(f"`{path}`")


def render_about_tab():
    st.write(
        "**ConflictGuard AI** is a multi-agent framework for detecting and "
        "resolving conflicting evidence in Retrieval-Augmented Generation "
        "systems, developed as a master's capstone project."
    )

    st.markdown(
        "1. **Retrieval** — semantic search over the document collection\n"
        "2. **Claim Extraction** — converts each passage into comparable factual claims\n"
        "3. **Conflict Detection** — classifies agreement, temporal, conditional, "
        "source-reliability, ambiguous, or insufficient-evidence cases\n"
        "4. **Source Evaluation** — scores relevance, recency, authority, and applicability\n"
        "5. **Resolution** — picks a conflict-specific strategy: prefer the newer or more "
        "authoritative source, explain conditional differences, present both positions, "
        "ask for clarification, or abstain\n"
        "6. **Verification** — checks that the final answer is actually supported by the "
        "selected evidence before it is returned\n"
    )

    st.write(
        "Conflict categories: direct factual conflict, temporal conflict, conditional "
        "conflict, source-reliability conflict, ambiguous question, insufficient evidence, "
        "and no conflict."
    )

    st.write(
        "The three configurations compared throughout this dashboard are a standard RAG "
        "baseline, a single-agent conflict-aware baseline, and the proposed multi-agent "
        "ConflictGuard pipeline."
    )


def main():
    st.title("ConflictGuard AI")
    st.caption("A multi-agent framework for detecting and resolving conflicting evidence in RAG systems")

    tab_query, tab_eval, tab_about = st.tabs(["Live Query", "Evaluation Results", "About"])

    with tab_query:
        render_live_query_tab()
    with tab_eval:
        render_evaluation_tab()
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
