import ollama

from app.rag.retriever import Retriever
from app.security.output_validator import redact_sensitive_data
from app.security.prompt_detector import PromptInjectionDetector


class RAGAssistant:
    def __init__(self, protected=False):
        self.retriever = Retriever()
        self.model = "qwen2.5:3b"
        self.protected = protected
        self.prompt_detector = PromptInjectionDetector()

    def answer(self, question):
        prompt_risk = {
            "is_attack": False,
            "risk_score": 0,
            "matched_patterns": []
        }

        # Input guardrail
        if self.protected:
            prompt_risk = self.prompt_detector.detect(question)

            if prompt_risk["is_attack"]:
                return {
                    "question": question,
                    "answer": "Request blocked by DeepShield because the prompt appears to contain a prompt-injection attempt.",
                    "raw_answer": "",
                    "sources": [],
                    "detected_sensitive_terms": [],
                    "prompt_risk": prompt_risk,
                    "blocked": True
                }

        results = self.retriever.search(
            question,
            n_results=1
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return {
                "question": question,
                "answer": "I do not have enough information to answer that.",
                "raw_answer": "",
                "sources": [],
                "detected_sensitive_terms": [],
                "prompt_risk": prompt_risk,
                "blocked": False
            }

        context = "\n\n".join(documents)

        prompt = f"""
You are an information extraction assistant.

Read the reference text and answer the question using only information explicitly
present in the reference text.

Do not interpret policies.
Do not make access-control decisions.
Do not add outside information.

If the answer exists in the reference text, state it directly and concisely.

REFERENCE TEXT:
{context}

QUESTION:
{question}

DIRECT ANSWER:
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        raw_answer = response["message"]["content"].strip()

        sources = [
            metadata["source"]
            for metadata in metadatas
            if metadata and "source" in metadata
        ]

        detected_sensitive_terms = []

        # Output guardrail
        if self.protected:
            final_answer, detected_sensitive_terms = redact_sensitive_data(
                raw_answer
            )
        else:
            final_answer = raw_answer

        return {
            "question": question,
            "answer": final_answer,
            "raw_answer": raw_answer,
            "sources": sources,
            "detected_sensitive_terms": detected_sensitive_terms,
            "prompt_risk": prompt_risk,
            "blocked": False
        }
