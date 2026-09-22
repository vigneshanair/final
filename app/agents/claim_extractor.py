import json
import re

import ollama


class ClaimExtractionAgent:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    def _extract_metadata(self, passage):

        year_match = re.search(
            r"Publication Year:\s*(\d{4})",
            passage,
            re.IGNORECASE
        )

        version_match = re.search(
            r"Policy Version:\s*([0-9.]+)",
            passage,
            re.IGNORECASE
        )

        authority_match = re.search(
            r"Source Authority:\s*([^\n]+)",
            passage,
            re.IGNORECASE
        )

        status_match = re.search(
            r"Document Status:\s*([^\n]+)",
            passage,
            re.IGNORECASE
        )

        publication_year = (
            year_match.group(1).strip()
            if year_match
            else None
        )

        policy_version = (
            version_match.group(1).strip()
            if version_match
            else None
        )

        source_authority = (
            authority_match.group(1).strip()
            if authority_match
            else None
        )

        document_status = (
            status_match.group(1).strip()
            if status_match
            else None
        )

        return (
            publication_year,
            policy_version,
            source_authority,
            document_status
        )

    def _extract_applicability(self, passage):

        patterns = [
            r"applies\s+only\s+to\s+([^.]+)",
            r"applies\s+to\s+([^.]+)"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                passage,
                re.IGNORECASE
            )

            if match:
                return (
                    match.group(1)
                    .strip()
                    .rstrip(".")
                )

        return None

    def extract(self, passage, source="unknown"):

        (
            publication_year,
            policy_version,
            source_authority,
            document_status
        ) = self._extract_metadata(passage)

        document_condition = (
            self._extract_applicability(passage)
        )

        prompt = f"""
You are a claim extraction agent.

Extract factual claims from the provided passage.

Rules:

1. Use only information explicitly stated in the passage.
2. Do not add outside knowledge.
3. Do not resolve conflicts.
4. Split separate factual statements into separate claims.
5. Preserve quantities, conditions, dates, and exceptions.
6. Preserve population/applicability conditions when present.
7. Return valid JSON only.

For each claim return:

- claim
- subject
- attribute
- value
- time_scope
- condition

Return:

{{
    "claims": [
        {{
            "claim": "...",
            "subject": "...",
            "attribute": "...",
            "value": "...",
            "time_scope": null,
            "condition": null
        }}
    ]
}}

PASSAGE:

{passage}
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
            response["message"]["content"].strip()
        )

        try:
            parsed = json.loads(raw_output)

            claims = parsed.get(
                "claims",
                []
            )

        except json.JSONDecodeError:
            claims = []

        if publication_year:

            for claim in claims:
                claim["time_scope"] = publication_year

        if document_condition:

            for claim in claims:

                if not claim.get("condition"):
                    claim["condition"] = (
                        document_condition
                    )

        return {
            "source": source,
            "publication_year": publication_year,
            "policy_version": policy_version,
            "source_authority": source_authority,
            "document_status": document_status,
            "document_condition": document_condition,
            "claims": claims,
            "raw_output": raw_output
        }
