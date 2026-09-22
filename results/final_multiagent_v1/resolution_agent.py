import json
import re

import ollama


class ResolutionAgent:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    # =================================================
    # RELEVANCE HELPERS
    # =================================================

    def _tokenize(self, text):
        return set(
            re.findall(
                r"[a-z0-9]+(?:-[a-z0-9]+)?",
                str(text).lower()
            )
        )

    def _content_tokens(self, text):
        stopwords = {
            "a", "an", "the",
            "is", "are", "was", "were",
            "be", "been", "being",
            "what", "which", "who",
            "how", "when", "where", "why",
            "do", "does", "did",
            "can", "could", "may", "might",
            "must", "should", "would", "will",
            "of", "to", "for", "from",
            "in", "on", "at", "by", "with",
            "and", "or",
            "this", "that", "these", "those",
            "their", "they", "them",
            "it", "its",
            "employee", "employees",
            "worker", "workers",
            "staff",
            "user", "users",
            "company",
            "policy",
            "document",
            "source"
        }

        return {
            token
            for token in self._tokenize(text)
            if token not in stopwords
        }

    def _relevance_score(
        self,
        question,
        claim
    ):
        question_tokens = self._content_tokens(
            question
        )

        claim_text = " ".join([
            str(claim.get("claim", "")),
            str(claim.get("subject", "")),
            str(claim.get("attribute", "")),
            str(claim.get("value", ""))
        ])

        claim_tokens = self._content_tokens(
            claim_text
        )

        return len(
            question_tokens
            & claim_tokens
        )

    def _best_claim_from_source(
        self,
        question,
        source_item
    ):
        best_claim = None
        best_score = -1

        for claim in source_item.get(
            "claims",
            []
        ):
            score = self._relevance_score(
                question,
                claim
            )

            if score > best_score:
                best_score = score
                best_claim = claim.get(
                    "claim"
                )

        return best_claim

    # =================================================
    # MAIN RESOLUTION
    # =================================================

    def resolve(
        self,
        question,
        evidence,
        conflict_result,
        source_evaluation
    ):

        conflict_type = conflict_result.get(
            "conflict_type"
        )

        preferred_source = source_evaluation.get(
            "preferred_source"
        )

        # =================================================
        # DIRECT FACTUAL CONFLICT
        # =================================================

        if (
            conflict_type == "direct_factual_conflict"
            and not preferred_source
        ):

            claims = conflict_result.get(
                "conflicting_claims",
                []
            )

            positions = []

            for claim in claims:

                source = claim.get(
                    "source",
                    "unknown source"
                )

                claim_text = claim.get(
                    "claim",
                    ""
                )

                if claim_text:
                    positions.append(
                        f"{source}: {claim_text}"
                    )

            if positions:
                answer = (
                    "The available sources disagree. "
                    + " ".join(positions)
                )
            else:
                answer = (
                    "The available sources contain "
                    "conflicting information, and there "
                    "is not enough evidence to determine "
                    "which position should be preferred."
                )

            return {
                "resolution_strategy": (
                    "present_both_positions"
                ),
                "selected_source": None,
                "selected_claim": None,
                "resolved_answer": answer,
                "decision_reason": (
                    "The sources provide conflicting "
                    "factual values and there is no "
                    "evidence establishing that one "
                    "source should be preferred."
                ),
                "abstain": False,
                "raw_output": ""
            }

        # =================================================
        # TEMPORAL CONFLICT
        # =================================================

        if (
            conflict_type == "temporal_conflict"
            and preferred_source
        ):

            selected_claim = None

            for item in evidence:

                if item.get("source") == preferred_source:

                    selected_claim = (
                        self._best_claim_from_source(
                            question,
                            item
                        )
                    )

                    break

            return {
                "resolution_strategy": (
                    "prefer_newer_authoritative_source"
                ),
                "selected_source": preferred_source,
                "selected_claim": selected_claim,
                "resolved_answer": (
                    selected_claim
                    if selected_claim
                    else (
                        "The preferred source should "
                        "be used to answer the question."
                    )
                ),
                "decision_reason": (
                    source_evaluation.get(
                        "preference_reason",
                        "The preferred source was selected "
                        "based on recency, version, or "
                        "supersession evidence."
                    )
                ),
                "abstain": False,
                "raw_output": ""
            }

        # =================================================
        # CONDITIONAL CONFLICT
        # =================================================

        if conflict_type == "conditional_conflict":

            relevant_positions = []

            for item in evidence:

                for claim in item.get(
                    "claims",
                    []
                ):

                    score = self._relevance_score(
                        question,
                        claim
                    )

                    condition = claim.get(
                        "condition"
                    )

                    claim_text = claim.get(
                        "claim"
                    )

                    if (
                        score > 0
                        and condition
                        and claim_text
                    ):
                        relevant_positions.append(
                            claim_text
                        )

            if relevant_positions:

                answer = " ".join(
                    relevant_positions
                )

            else:

                answer = (
                    "The available evidence applies "
                    "under different conditions, so "
                    "the answer depends on which "
                    "condition applies."
                )

            return {
                "resolution_strategy": (
                    "explain_conditional_difference"
                ),
                "selected_source": None,
                "selected_claim": None,
                "resolved_answer": answer,
                "decision_reason": (
                    "The differing claims apply to "
                    "different populations or conditions, "
                    "so both may be valid within their "
                    "respective contexts."
                ),
                "abstain": False,
                "raw_output": ""
            }

        # =================================================
        # SOURCE RELIABILITY CONFLICT
        # =================================================

        if (
            conflict_type
            == "source_reliability_conflict"
            and preferred_source
        ):

            selected_claim = None

            for item in evidence:

                if item.get("source") == preferred_source:

                    selected_claim = (
                        self._best_claim_from_source(
                            question,
                            item
                        )
                    )

                    break

            return {
                "resolution_strategy": (
                    "prefer_authoritative_source"
                ),
                "selected_source": preferred_source,
                "selected_claim": selected_claim,
                "resolved_answer": (
                    selected_claim
                    if selected_claim
                    else (
                        "The authoritative source "
                        "should be preferred."
                    )
                ),
                "decision_reason": (
                    source_evaluation.get(
                        "preference_reason",
                        "The selected source has stronger "
                        "explicit authority or document "
                        "status."
                    )
                ),
                "abstain": False,
                "raw_output": ""
            }

        # =================================================
        # AMBIGUOUS QUESTION
        # =================================================

        if conflict_type == "ambiguous_question":

            conditions = []

            for item in evidence:

                condition = item.get(
                    "document_condition"
                )

                if condition:

                    normalized = (
                        str(condition)
                        .strip()
                    )

                    if normalized not in conditions:
                        conditions.append(
                            normalized
                        )

            if len(conditions) >= 2:

                clarification = (
                    "Do you mean "
                    + " or ".join(conditions)
                    + "?"
                )

            else:

                clarification = (
                    "Could you clarify which condition "
                    "or population you mean?"
                )

            return {
                "resolution_strategy": (
                    "ask_for_clarification"
                ),
                "selected_source": None,
                "selected_claim": None,
                "resolved_answer": clarification,
                "decision_reason": (
                    "The evidence contains different "
                    "answers for different populations "
                    "or conditions, but the question "
                    "does not specify which one applies."
                ),
                "abstain": False,
                "raw_output": ""
            }

        # =================================================
        # INSUFFICIENT EVIDENCE
        # =================================================

        if conflict_type == "insufficient_evidence":

            return {
                "resolution_strategy": "abstain",
                "selected_source": None,
                "selected_claim": None,
                "resolved_answer": (
                    "I do not have enough evidence "
                    "to answer this question reliably."
                ),
                "decision_reason": (
                    "The retrieved evidence does not "
                    "contain sufficient information to "
                    "support a reliable answer."
                ),
                "abstain": True,
                "raw_output": ""
            }

        # =================================================
        # NO CONFLICT
        # =================================================

        if conflict_type == "no_conflict":

            selected_source = None
            selected_claim = None
            best_score = -1

            for item in evidence:

                for claim in item.get(
                    "claims",
                    []
                ):

                    score = self._relevance_score(
                        question,
                        claim
                    )

                    if score > best_score:

                        best_score = score

                        selected_source = (
                            item.get("source")
                        )

                        selected_claim = (
                            claim.get("claim")
                        )

            return {
                "resolution_strategy": (
                    "no_resolution_needed"
                ),
                "selected_source": selected_source,
                "selected_claim": selected_claim,
                "resolved_answer": (
                    selected_claim
                    if selected_claim
                    else (
                        "The available evidence does "
                        "not contain a conflict."
                    )
                ),
                "decision_reason": (
                    "The relevant evidence is consistent, "
                    "so no conflict-resolution strategy "
                    "is required."
                ),
                "abstain": False,
                "raw_output": ""
            }

        # =================================================
        # LLM FALLBACK
        # =================================================

        prompt = f"""
You are a resolution agent in a conflict-aware
Retrieval-Augmented Generation system.

QUESTION:
{question}

EVIDENCE:
{json.dumps(evidence, indent=2)}

CONFLICT RESULT:
{json.dumps(conflict_result, indent=2)}

SOURCE EVALUATION:
{json.dumps(source_evaluation, indent=2)}

Possible strategies:

- prefer_newer_authoritative_source
- prefer_authoritative_source
- explain_conditional_difference
- present_both_positions
- ask_for_clarification
- abstain
- no_resolution_needed

Rules:

1. Use only supplied evidence.
2. Do not invent facts.
3. If one policy explicitly supersedes another,
   prefer the superseding policy.
4. If one source is explicitly more authoritative,
   prefer the authoritative source.
5. If conflicting claims apply under different
   conditions, explain those conditions.
6. If equally supported sources directly disagree,
   present both positions.
7. If the question is ambiguous, ask for clarification.
8. If evidence is insufficient, abstain.
9. If relevant evidence agrees, no conflict resolution
   is needed.
10. Return valid JSON only.

Return:

{{
    "resolution_strategy": "...",
    "selected_source": null,
    "selected_claim": null,
    "resolved_answer": "...",
    "decision_reason": "...",
    "abstain": false
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
                "resolution_strategy": "abstain",
                "selected_source": None,
                "selected_claim": None,
                "resolved_answer": (
                    "I do not have enough reliable "
                    "information to resolve the conflict."
                ),
                "decision_reason": (
                    "Resolution agent returned invalid JSON."
                ),
                "abstain": True
            }

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
            "resolved_answer",
            ""
        )

        result.setdefault(
            "decision_reason",
            ""
        )

        result.setdefault(
            "abstain",
            False
        )

        if (
            result["resolution_strategy"]
            == "present_both_positions"
        ):
            result["selected_source"] = None
            result["selected_claim"] = None

        if (
            result["resolution_strategy"]
            == "ask_for_clarification"
        ):
            result["selected_source"] = None
            result["selected_claim"] = None

        if (
            result["resolution_strategy"]
            == "abstain"
        ):
            result["selected_source"] = None
            result["selected_claim"] = None
            result["abstain"] = True

        result["raw_output"] = raw_output

        return result
