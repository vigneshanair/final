import json

import ollama


class SourceEvaluationAgent:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    def evaluate(self, question, evidence, conflict_result):

        prompt = f"""
You are a source evaluation agent in a conflict-aware
Retrieval-Augmented Generation system.

Your job is to evaluate the sources involved in the detected conflict.

QUESTION:
{question}

EVIDENCE:
{json.dumps(evidence, indent=2)}

CONFLICT RESULT:
{json.dumps(conflict_result, indent=2)}

Evaluate each source using only the supplied evidence.

Consider:

1. Relevance to the user's question
2. Publication year
3. Policy version
4. Whether one source explicitly supersedes another
5. Authority information explicitly stated in the evidence
6. Applicability to the user's question

Important rules:

- Use only information contained in the supplied evidence.
- Do not invent authority information.
- Do not automatically prefer a source only because it is newer.
- Explicit supersession is strong evidence that one policy replaces another.
- Do not generate the final user answer.
- Do not resolve anything beyond selecting the most appropriate source.
- Return valid JSON only.

Return exactly this structure:

{{
    "source_evaluations": [
        {{
            "source": "...",
            "relevance": "high",
            "publication_year": "...",
            "policy_version": "...",
            "supersedes_other_source": false,
            "superseded_source": null,
            "applicability": "...",
            "evaluation_reason": "..."
        }}
    ],
    "preferred_source": "...",
    "preference_reason": "..."
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

        raw_output = response["message"]["content"].strip()

        try:
            result = json.loads(raw_output)

        except json.JSONDecodeError:
            result = {
                "source_evaluations": [],
                "preferred_source": None,
                "preference_reason": (
                    "Source evaluator returned invalid JSON."
                )
            }

        result.setdefault(
            "source_evaluations",
            []
        )

        result.setdefault(
            "preferred_source",
            None
        )

        result.setdefault(
            "preference_reason",
            ""
        )

        result["raw_output"] = raw_output

        return result
