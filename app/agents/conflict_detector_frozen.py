import json
import re

import ollama


class ConflictDetectionAgent:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    # =====================================================
    # RELEVANCE
    # =====================================================

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

        question_tokens = (
            self._content_tokens(
                question
            )
        )

        claim_text = " ".join([
            str(
                claim.get(
                    "claim",
                    ""
                )
            ),
            str(
                claim.get(
                    "subject",
                    ""
                )
            ),
            str(
                claim.get(
                    "attribute",
                    ""
                )
            ),
            str(
                claim.get(
                    "value",
                    ""
                )
            )
        ])

        claim_tokens = (
            self._content_tokens(
                claim_text
            )
        )

        overlap = (
            question_tokens
            & claim_tokens
        )

        return len(overlap)

    def _get_top_relevant_claims(
        self,
        question,
        evidence
    ):
        """
        Select the most relevant claim(s) from each source.

        This prevents secondary claims such as manager
        approval from interfering with a question about
        remote-work frequency, and vice versa.
        """

        relevant = []

        for item in evidence:

            source = item.get(
                "source"
            )

            scored = []

            for claim in item.get(
                "claims",
                []
            ):

                score = (
                    self._relevance_score(
                        question,
                        claim
                    )
                )

                if score > 0:
                    scored.append(
                        (
                            score,
                            claim
                        )
                    )

            if not scored:
                continue

            max_score = max(
                score
                for score, _ in scored
            )

            for score, claim in scored:

                if score == max_score:

                    relevant.append({
                        "source": source,
                        "score": score,
                        "claim": claim
                    })

        return relevant

    def _has_relevant_evidence(
        self,
        question,
        evidence
    ):

        return len(
            self._get_top_relevant_claims(
                question,
                evidence
            )
        ) > 0

    # =====================================================
    # NO-CONFLICT AGREEMENT
    # =====================================================

    def _normalize_claim_text(
        self,
        text
    ):

        text = str(text).lower()

        text = re.sub(
            r"[^a-z0-9\s-]",
            "",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    def _relevant_claims_agree(
        self,
        question,
        evidence
    ):

        relevant = (
            self._get_top_relevant_claims(
                question,
                evidence
            )
        )

        if len(relevant) < 2:
            return False

        sources = {
            item["source"]
            for item in relevant
        }

        if len(sources) < 2:
            return False

        normalized_claims = {
            self._normalize_claim_text(
                item["claim"].get(
                    "claim",
                    ""
                )
            )
            for item in relevant
            if item["claim"].get(
                "claim"
            )
        }

        return (
            len(normalized_claims) == 1
        )

    # =====================================================
    # TEMPORAL EVIDENCE
    # =====================================================

    def _has_temporal_evidence(
        self,
        evidence
    ):

        years = set()
        versions = set()
        supersession_found = False

        for item in evidence:

            year = item.get(
                "publication_year"
            )

            if year:
                years.add(
                    str(year)
                )

            version = item.get(
                "policy_version"
            )

            if version:
                versions.add(
                    str(version)
                )

            for claim in item.get(
                "claims",
                []
            ):

                time_scope = claim.get(
                    "time_scope"
                )

                if time_scope:
                    years.add(
                        str(time_scope)
                    )

                claim_text = str(
                    claim.get(
                        "claim",
                        ""
                    )
                ).lower()

                if "supersede" in claim_text:
                    supersession_found = True

        return (
            len(years) > 1
            or len(versions) > 1
            or supersession_found
        )

    # =====================================================
    # CONDITIONAL EVIDENCE
    # =====================================================

    def _get_conditions(
        self,
        evidence
    ):

        conditions = set()

        for item in evidence:

            document_condition = (
                item.get(
                    "document_condition"
                )
            )

            if document_condition:

                conditions.add(
                    str(document_condition)
                    .strip()
                    .lower()
                )

            for claim in item.get(
                "claims",
                []
            ):

                condition = claim.get(
                    "condition"
                )

                if condition:

                    conditions.add(
                        str(condition)
                        .strip()
                        .lower()
                    )

        return conditions

    def _has_conditional_evidence(
        self,
        evidence
    ):

        return (
            len(
                self._get_conditions(
                    evidence
                )
            )
            > 1
        )

    # =====================================================
    # SOURCE RELIABILITY
    # =====================================================

    def _has_reliability_evidence(
        self,
        evidence
    ):

        authority_signals = []

        for item in evidence:

            authority = item.get(
                "source_authority"
            )

            status = item.get(
                "document_status"
            )

            combined = " ".join(
                str(value).lower()
                for value in [
                    authority,
                    status
                ]
                if value
            )

            if combined:
                authority_signals.append(
                    combined
                )

        if len(authority_signals) < 2:
            return False

        official_found = any(
            (
                "official" in signal
                or "human resources" in signal
                or "hr department" in signal
            )
            and "unofficial" not in signal
            for signal in authority_signals
        )

        unofficial_found = any(
            (
                "unofficial" in signal
                or "informal" in signal
                or "employee notes" in signal
            )
            for signal in authority_signals
        )

        return (
            official_found
            and unofficial_found
        )

    # =====================================================
    # AMBIGUITY
    # =====================================================

    def _condition_keywords(
        self,
        condition
    ):

        generic_words = {
            "employee",
            "employees",
            "worker",
            "workers",
            "staff",
            "person",
            "people",
            "user",
            "users",
            "the",
            "a",
            "an",
            "to",
            "for",
            "only"
        }

        tokens = re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)?",
            str(condition).lower()
        )

        return {
            token
            for token in tokens
            if token not in generic_words
        }

    def _question_condition_matches(
        self,
        question,
        evidence
    ):

        question_lower = (
            question.lower()
        )

        conditions = (
            self._get_conditions(
                evidence
            )
        )

        matched = []

        for condition in conditions:

            keywords = (
                self._condition_keywords(
                    condition
                )
            )

            if any(
                keyword in question_lower
                for keyword in keywords
            ):

                matched.append(
                    condition
                )

        return matched

    def _is_ambiguous_question(
        self,
        question,
        evidence
    ):

        conditions = (
            self._get_conditions(
                evidence
            )
        )

        if len(conditions) <= 1:
            return False

        matched = (
            self._question_condition_matches(
                question,
                evidence
            )
        )

        return len(matched) == 0

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize_conflict_type(
        self,
        result,
        question,
        evidence
    ):

        conflict_type = result.get(
            "conflict_type",
            "insufficient_evidence"
        )

        # -------------------------------------------------
        # 1. INSUFFICIENT EVIDENCE
        # -------------------------------------------------

        if not self._has_relevant_evidence(
            question,
            evidence
        ):

            result["conflict_detected"] = False

            result["conflict_type"] = (
                "insufficient_evidence"
            )

            result["conflicting_claims"] = []

            result["explanation"] = (
                "The supplied evidence does not contain "
                "claims relevant to the question. "
                "Therefore there is insufficient evidence "
                "to provide a reliable answer."
            )

            return result

        # -------------------------------------------------
        # 2. NO CONFLICT
        # -------------------------------------------------

        if self._relevant_claims_agree(
            question,
            evidence
        ):

            result["conflict_detected"] = False

            result["conflict_type"] = (
                "no_conflict"
            )

            result["conflicting_claims"] = []

            result["explanation"] = (
                "The most relevant claims from the "
                "available sources agree with each other. "
                "Therefore no conflict is present."
            )

            return result

        temporal_evidence = (
            self._has_temporal_evidence(
                evidence
            )
        )

        conditional_evidence = (
            self._has_conditional_evidence(
                evidence
            )
        )

        reliability_evidence = (
            self._has_reliability_evidence(
                evidence
            )
        )

        ambiguous_question = (
            self._is_ambiguous_question(
                question,
                evidence
            )
        )

        recognized = {
            "direct_factual_conflict",
            "temporal_conflict",
            "conditional_conflict",
            "source_reliability_conflict",
            "ambiguous_question",
            "no_conflict",
            "insufficient_evidence"
        }

        # -------------------------------------------------
        # 3. AMBIGUOUS QUESTION
        # -------------------------------------------------

        if (
            ambiguous_question
            and conditional_evidence
            and conflict_type in recognized
        ):

            conditions = sorted(
                self._get_conditions(
                    evidence
                )
            )

            result["conflict_detected"] = True

            result["conflict_type"] = (
                "ambiguous_question"
            )

            result["explanation"] = (
                "The evidence contains different answers "
                "for different conditions or populations: "
                + ", ".join(conditions)
                + ". The question does not specify which "
                "condition or population is intended, so "
                "clarification is required."
            )

            return result

        # -------------------------------------------------
        # 4. SOURCE RELIABILITY
        # -------------------------------------------------

        if (
            reliability_evidence
            and not temporal_evidence
            and not conditional_evidence
        ):

            result["conflict_detected"] = True

            result["conflict_type"] = (
                "source_reliability_conflict"
            )

            result["explanation"] = (
                "The sources provide incompatible factual "
                "values and explicitly differ in authority "
                "or document status."
            )

            return result

        # -------------------------------------------------
        # 5. TEMPORAL
        # -------------------------------------------------

        if (
            temporal_evidence
            and not conditional_evidence
        ):

            result["conflict_detected"] = True

            result["conflict_type"] = (
                "temporal_conflict"
            )

            result["explanation"] = (
                "The relevant sources contain different "
                "values and explicit temporal evidence "
                "such as different policy years, versions, "
                "or supersession."
            )

            return result

        # -------------------------------------------------
        # 6. CONDITIONAL
        # -------------------------------------------------

        if (
            conditional_evidence
            and not temporal_evidence
            and not ambiguous_question
        ):

            conditions = sorted(
                self._get_conditions(
                    evidence
                )
            )

            result["conflict_detected"] = True

            result["conflict_type"] = (
                "conditional_conflict"
            )

            result["explanation"] = (
                "The relevant claims apply under "
                "different explicit conditions or "
                "populations: "
                + ", ".join(conditions)
                + "."
            )

            return result

        # -------------------------------------------------
        # -------------------------------------------------
        # 7. DIRECT FACTUAL CONFLICT
        # -------------------------------------------------
        #
        # If we reached this point:
        #
        # - relevant evidence exists
        # - relevant claims do NOT agree
        # - it is not ambiguous
        # - it is not temporal
        # - it is not conditional
        # - it is not source-reliability based
        #
        # Therefore the disagreement is a direct
        # factual conflict, even if the LLM incorrectly
        # returned "no_conflict".
        # -------------------------------------------------

        result["conflict_detected"] = True

        result["conflict_type"] = (
            "direct_factual_conflict"
        )

        result["explanation"] = (
            "The relevant sources provide incompatible "
            "factual values, and there is no explicit "
            "temporal, conditional, or source-authority "
            "difference that explains the disagreement. "
            "Therefore this is a direct factual conflict."
        )

        return result
    # =====================================================
    # MAIN DETECTOR
    # =====================================================

    def detect(
        self,
        question,
        evidence
    ):

        prompt = f"""
You are a conflict detection agent for a
Retrieval-Augmented Generation system.

QUESTION:
{question}

EVIDENCE:
{json.dumps(evidence, indent=2)}

Choose exactly one:

- direct_factual_conflict
- temporal_conflict
- conditional_conflict
- source_reliability_conflict
- ambiguous_question
- insufficient_evidence
- no_conflict

Rules:

1. First determine whether relevant evidence exists.
2. If relevant evidence is missing, use
   insufficient_evidence.
3. If relevant claims agree, use no_conflict.
4. Different policy years, versions, or supersession
   indicate temporal conflict.
5. Different populations or conditions may indicate
   conditional conflict.
6. If multiple relevant populations exist but the
   question does not specify which one is intended,
   use ambiguous_question.
7. Explicit official versus unofficial status may
   indicate source_reliability_conflict.
8. Do not invent facts.
9. Do not resolve the disagreement.
10. Return valid JSON only.

Return:

{{
    "conflict_detected": false,
    "conflict_type": "no_conflict",
    "conflicting_claims": [],
    "explanation": "...",
    "confidence": null
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
                "conflicting_claims": [],
                "explanation": (
                    "Conflict detector returned "
                    "invalid JSON."
                ),
                "confidence": None
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
            "conflicting_claims",
            []
        )

        result.setdefault(
            "explanation",
            ""
        )

        result = (
            self._normalize_conflict_type(
                result,
                question,
                evidence
            )
        )

        confidence = result.get(
            "confidence"
        )

        try:

            if confidence is not None:
                confidence = float(
                    confidence
                )

        except (
            TypeError,
            ValueError
        ):

            confidence = None

        result["confidence"] = confidence
        result["raw_output"] = raw_output

        return result
