from app.rag.retriever import Retriever

from app.agents.claim_extractor import ClaimExtractionAgent
from app.agents.conflict_detector import ConflictDetectionAgent
from app.agents.source_evaluator import SourceEvaluationAgent
from app.agents.resolution_agent import ResolutionAgent
from app.agents.verification_agent import VerificationAgent


class ConflictGuardCoordinator:

    def __init__(self):
        self.retriever = Retriever()
        self.claim_agent = ClaimExtractionAgent()
        self.conflict_agent = ConflictDetectionAgent()
        self.source_agent = SourceEvaluationAgent()
        self.resolution_agent = ResolutionAgent()
        self.verification_agent = VerificationAgent()

    def answer(self, question, n_results=3):

        # -------------------------------------------------
        # STEP 1: RETRIEVAL
        # -------------------------------------------------

        results = self.retriever.search(
            question,
            n_results=n_results
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        if not documents:
            return {
                "question": question,
                "final_answer": (
                    "I do not have enough information "
                    "to answer this question."
                ),
                "abstained": True,
                "conflict_detected": False,
                "conflict_type": "insufficient_evidence",
                "evidence": [],
                "source_evaluation": {},
                "resolution": {},
                "verification": {}
            }

        # -------------------------------------------------
        # STEP 2: CLAIM EXTRACTION
        # -------------------------------------------------

        evidence = []

        for index, document in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            source = metadata.get(
                "source",
                f"document_{index + 1}"
            )

            extracted = self.claim_agent.extract(
                document,
                source=source
            )

            evidence.append({
                "source": source,
                "publication_year": (
                    extracted.get("publication_year")
                ),
                "policy_version": (
                    extracted.get("policy_version")
                ),
                "claims": (
                    extracted.get("claims", [])
                )
            })

        # -------------------------------------------------
        # STEP 3: CONFLICT DETECTION
        # -------------------------------------------------

        conflict_result = self.conflict_agent.detect(
            question,
            evidence
        )

        # -------------------------------------------------
        # STEP 4: SOURCE EVALUATION
        # -------------------------------------------------

        source_result = self.source_agent.evaluate(
            question,
            evidence,
            conflict_result
        )

        # -------------------------------------------------
        # STEP 5: RESOLUTION
        # -------------------------------------------------

        resolution_result = (
            self.resolution_agent.resolve(
                question,
                evidence,
                conflict_result,
                source_result
            )
        )

        # -------------------------------------------------
        # STEP 6: VERIFICATION
        # -------------------------------------------------

        verification_result = (
            self.verification_agent.verify(
                question,
                evidence,
                resolution_result
            )
        )

        # -------------------------------------------------
        # STEP 7: FINAL DECISION
        # -------------------------------------------------

        abstained = bool(
            resolution_result.get(
                "abstain",
                False
            )
        )

        verified = bool(
            verification_result.get(
                "verified",
                False
            )
        )

        if abstained:
            final_answer = (
                resolution_result.get(
                    "resolved_answer"
                )
                or
                "I cannot reliably resolve this question "
                "from the available evidence."
            )

        elif not verified:
            final_answer = (
                "ConflictGuard could not verify the "
                "resolved answer against the evidence."
            )

            abstained = True

        else:
            final_answer = (
                resolution_result.get(
                    "resolved_answer",
                    ""
                )
            )

        # -------------------------------------------------
        # FINAL STRUCTURED RESULT
        # -------------------------------------------------

        return {
            "question": question,

            "final_answer": final_answer,

            "abstained": abstained,

            "conflict_detected": (
                conflict_result.get(
                    "conflict_detected",
                    False
                )
            ),

            "conflict_type": (
                conflict_result.get(
                    "conflict_type"
                )
            ),

            "conflict_explanation": (
                conflict_result.get(
                    "explanation"
                )
            ),

            "evidence": evidence,

            "source_evaluation": source_result,

            "resolution": resolution_result,

            "verification": verification_result,

            "selected_source": (
                resolution_result.get(
                    "selected_source"
                )
            ),

            "resolution_strategy": (
                resolution_result.get(
                    "resolution_strategy"
                )
            ),

            "verified": verified
        }
