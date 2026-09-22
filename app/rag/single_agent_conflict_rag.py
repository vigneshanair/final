import json

import ollama


class SingleAgentConflictRAG:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    def answer(
        self,
        question,
        evidence
    ):

        prompt = f"""
You are a single-agent conflict-aware
Retrieval-Augmented Generation system.

You must inspect all supplied evidence and produce
one final structured decision.

QUESTION:
{question}

EVIDENCE:
{json.dumps(evidence, indent=2)}

Choose exactly one conflict type:

- direct_factual_conflict
- temporal_conflict
- conditional_conflict
- source_reliability_conflict
- ambiguous_question
- insufficient_evidence
- no_conflict

Choose exactly one resolution strategy:

- prefer_newer_authoritative_source
- prefer_authoritative_source
- explain_conditional_difference
- present_both_positions
- ask_for_clarification
- abstain
- no_resolution_needed

Definitions:

direct_factual_conflict:
Relevant sources provide incompatible factual values
without a clear temporal, conditional, or authority
difference.

temporal_conflict:
Relevant claims disagree because of different years,
versions, dates, or explicit supersession.

conditional_conflict:
Different answers are valid under different explicit
populations, situations, or conditions, and the
question identifies those conditions.

source_reliability_conflict:
Relevant sources disagree and one source has stronger
explicit authority or official status.

ambiguous_question:
Multiple answers may apply under different conditions,
but the user's question does not specify which
condition or population is intended.

insufficient_evidence:
The evidence does not contain enough relevant
information to answer the question.

no_conflict:
Relevant claims agree.

Resolution rules:

1. Use only supplied evidence.
2. Do not invent facts.
3. If a newer policy explicitly supersedes an older
   policy, prefer the newer policy.
4. If one source is explicitly official and another
   is unofficial, prefer the official source.
5. If different answers apply under different explicit
   conditions, explain those differences.
6. If equally supported sources directly disagree,
   present both positions.
7. If the question is ambiguous, ask for clarification.
8. If evidence is insufficient, abstain.
9. If relevant evidence agrees, no resolution is needed.
10. Return valid JSON only.

Return exactly this structure:

{{
    "conflict_detected": true,
    "conflict_type": "...",
    "resolution_strategy": "...",
    "selected_source": null,
    "selected_claim": null,
    "answer": "...",
    "abstain": false,
    "explanation": "..."
}}
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={
                "temperature": 0
            }
        )

        raw_output = (
            response["message"]["content"]
            .strip()
        )

        try:
            result = json.loads(
                raw_output
            )

        except json.JSONDecodeError:
            result = {
                "conflict_detected": False,
                "conflict_type": (
                    "insufficient_evidence"
                ),
                "resolution_strategy": (
                    "abstain"
                ),
                "selected_source": None,
                "selected_claim": None,
                "answer": (
                    "I do not have enough reliable "
                    "information to answer this question."
                ),
                "abstain": True,
                "explanation": (
                    "The single-agent model returned "
                    "invalid JSON."
                )
            }

        result.setdefault(
            "conflict_detected",
            False
        )

        result.setdefault(
            "conflict_type",
            "insufficient_evidence"
        )

        result.setdefault(
            "resolution_strategy",
            "abstain"
        )

        result.setdefault(
            "selected_source",
            None
        )

        result.setdefault(
            "selected_claim",
            None
        )

        result.setdefault(
            "answer",
            ""
        )

        result.setdefault(
            "abstain",
            False
        )

        result.setdefault(
            "explanation",
            ""
        )

        result["raw_output"] = (
            raw_output
        )

        return result
