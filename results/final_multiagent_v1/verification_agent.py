import json

import ollama


class VerificationAgent:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    def _find_source(
        self,
        source_name,
        evidence
    ):
        if not source_name:
            return None

        for item in evidence:

            if item.get("source") == source_name:
                return item

        return None

    def _claim_exists(
        self,
        selected_claim,
        source_item
    ):
        if (
            not selected_claim
            or not source_item
        ):
            return False

        selected_normalized = (
            str(selected_claim)
            .strip()
            .lower()
        )

        for claim in source_item.get(
            "claims",
            []
        ):

            claim_text = (
                str(
                    claim.get(
                        "claim",
                        ""
                    )
                )
                .strip()
                .lower()
            )

            if (
                selected_normalized == claim_text
                or selected_normalized in claim_text
                or claim_text in selected_normalized
            ):
                return True

        return False

    def verify(
        self,
        question,
        evidence,
        resolution_result
    ):

        strategy = resolution_result.get(
            "resolution_strategy"
        )

        selected_source = (
            resolution_result.get(
                "selected_source"
            )
        )

        selected_claim = (
            resolution_result.get(
                "selected_claim"
            )
        )

        resolved_answer = (
            resolution_result.get(
                "resolved_answer",
                ""
            )
        )

        abstain = resolution_result.get(
            "abstain",
            False
        )

        # =================================================
        # AMBIGUOUS QUESTION / CLARIFICATION
        # =================================================

        if strategy == "ask_for_clarification":

            valid = (
                bool(resolved_answer)
                and selected_source is None
                and selected_claim is None
                and not abstain
            )

            return {
                "verified": valid,
                "selected_source_found": True,
                "claim_supported": True,
                "answer_supported": valid,
                "unsupported_information": False,
                "final_status": (
                    "verified_clarification"
                    if valid
                    else "verification_failed"
                ),
                "verification_reason": (
                    "The system correctly avoids choosing "
                    "a source or factual claim when the "
                    "question is ambiguous and instead "
                    "requests clarification."
                    if valid
                    else (
                        "The clarification response is "
                        "not structured correctly."
                    )
                ),
                "raw_output": ""
            }

        # =================================================
        # ABSTENTION
        # =================================================

        if strategy == "abstain":

            valid = (
                abstain
                and selected_source is None
                and selected_claim is None
                and bool(resolved_answer)
            )

            return {
                "verified": valid,
                "selected_source_found": True,
                "claim_supported": True,
                "answer_supported": valid,
                "unsupported_information": False,
                "final_status": (
                    "verified_abstention"
                    if valid
                    else "verification_failed"
                ),
                "verification_reason": (
                    "The system correctly abstains because "
                    "there is insufficient evidence and "
                    "does not select an unsupported source "
                    "or claim."
                    if valid
                    else (
                        "The abstention response is "
                        "not structured correctly."
                    )
                ),
                "raw_output": ""
            }

        # =================================================
        # SELECTED-SOURCE RESOLUTIONS
        # =================================================

        if strategy in {
            "prefer_newer_authoritative_source",
            "prefer_authoritative_source"
        }:

            source_item = self._find_source(
                selected_source,
                evidence
            )

            source_found = (
                source_item is not None
            )

            claim_supported = (
                self._claim_exists(
                    selected_claim,
                    source_item
                )
            )

            answer_supported = (
                bool(resolved_answer)
                and claim_supported
            )

            verified = (
                source_found
                and claim_supported
                and answer_supported
            )

            return {
                "verified": verified,
                "selected_source_found": (
                    source_found
                ),
                "claim_supported": (
                    claim_supported
                ),
                "answer_supported": (
                    answer_supported
                ),
                "unsupported_information": (
                    not verified
                ),
                "final_status": (
                    "verified"
                    if verified
                    else "verification_failed"
                ),
                "verification_reason": (
                    "The selected source exists in the "
                    "retrieved evidence, the selected claim "
                    "is supported by that source, and the "
                    "resolved answer is based on that claim."
                    if verified
                    else (
                        "The selected source, claim, or "
                        "resolved answer could not be fully "
                        "verified against the evidence."
                    )
                ),
                "raw_output": ""
            }

        # =================================================
        # LLM VERIFICATION FOR OTHER STRATEGIES
        # =================================================

        prompt = f"""
You are a verification agent in a conflict-aware
Retrieval-Augmented Generation system.

Your task is to determine whether the resolution
is supported by the supplied evidence.

QUESTION:
{question}

EVIDENCE:
{json.dumps(evidence, indent=2)}

RESOLUTION:
{json.dumps(resolution_result, indent=2)}

Check:

1. Does the answer use only information supported
   by the supplied evidence?

2. Does it avoid introducing unsupported facts?

3. If multiple positions are presented, are those
   positions actually present in the evidence?

4. If different conditions are explained, are those
   conditions explicitly supported by the evidence?

5. If a source or claim is selected, does it exist
   in the evidence?

Return valid JSON only.

Return:

{{
    "verified": true,
    "selected_source_found": true,
    "claim_supported": true,
    "answer_supported": true,
    "unsupported_information": false,
    "final_status": "verified",
    "verification_reason": "..."
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
                "verified": False,
                "selected_source_found": False,
                "claim_supported": False,
                "answer_supported": False,
                "unsupported_information": True,
                "final_status": (
                    "verification_failed"
                ),
                "verification_reason": (
                    "Verification agent returned "
                    "invalid JSON."
                )
            }

        result.setdefault(
            "verified",
            False
        )

        result.setdefault(
            "selected_source_found",
            False
        )

        result.setdefault(
            "claim_supported",
            False
        )

        result.setdefault(
            "answer_supported",
            False
        )

        result.setdefault(
            "unsupported_information",
            True
        )

        result.setdefault(
            "final_status",
            "verification_failed"
        )

        result.setdefault(
            "verification_reason",
            ""
        )

        result["raw_output"] = raw_output

        return result
