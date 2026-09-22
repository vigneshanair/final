from app.rag.retriever import Retriever

from app.agents.claim_extractor import ClaimExtractionAgent
from app.agents.conflict_detector import ConflictDetectionAgent
from app.agents.source_evaluator import SourceEvaluationAgent
from app.agents.resolution_agent import ResolutionAgent
from app.agents.verification_agent import VerificationAgent


retriever = Retriever()

claim_agent = ClaimExtractionAgent()

conflict_agent = ConflictDetectionAgent()

source_agent = SourceEvaluationAgent()

resolution_agent = ResolutionAgent()

verification_agent = VerificationAgent()


question = (
    "How many days per week can employees work remotely?"
)


results = retriever.search(
    question,
    n_results=2
)


documents = results.get(
    "documents",
    [[]]
)[0]

metadatas = results.get(
    "metadatas",
    [[]]
)[0]


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

    extracted = claim_agent.extract(
        document,
        source=source
    )

    evidence.append({
        "source": source,
        "publication_year": (
            extracted["publication_year"]
        ),
        "policy_version": (
            extracted["policy_version"]
        ),
        "claims": extracted["claims"]
    })


conflict_result = conflict_agent.detect(
    question,
    evidence
)


source_result = source_agent.evaluate(
    question,
    evidence,
    conflict_result
)


resolution_result = resolution_agent.resolve(
    question,
    evidence,
    conflict_result,
    source_result
)


verification_result = verification_agent.verify(
    question,
    evidence,
    resolution_result
)


print("\n" + "=" * 70)

print("VERIFICATION RESULT")

print("=" * 70)


print(
    "Verified:",
    verification_result.get(
        "verified"
    )
)

print(
    "Selected Source Found:",
    verification_result.get(
        "selected_source_found"
    )
)

print(
    "Claim Supported:",
    verification_result.get(
        "claim_supported"
    )
)

print(
    "Answer Supported:",
    verification_result.get(
        "answer_supported"
    )
)

print(
    "Unsupported Information:",
    verification_result.get(
        "unsupported_information"
    )
)

print(
    "Final Status:",
    verification_result.get(
        "final_status"
    )
)

print(
    "Verification Reason:",
    verification_result.get(
        "verification_reason"
    )
)
