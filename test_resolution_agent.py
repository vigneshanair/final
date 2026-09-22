from app.rag.retriever import Retriever

from app.agents.claim_extractor import ClaimExtractionAgent
from app.agents.conflict_detector import ConflictDetectionAgent
from app.agents.source_evaluator import SourceEvaluationAgent
from app.agents.resolution_agent import ResolutionAgent


retriever = Retriever()

claim_agent = ClaimExtractionAgent()

conflict_agent = ConflictDetectionAgent()

source_agent = SourceEvaluationAgent()

resolution_agent = ResolutionAgent()


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


print("\n" + "=" * 70)

print("RESOLUTION RESULT")

print("=" * 70)


print(
    "Strategy:",
    resolution_result.get(
        "resolution_strategy"
    )
)

print(
    "Selected Source:",
    resolution_result.get(
        "selected_source"
    )
)

print(
    "Selected Claim:",
    resolution_result.get(
        "selected_claim"
    )
)

print(
    "Resolved Answer:",
    resolution_result.get(
        "resolved_answer"
    )
)

print(
    "Decision Reason:",
    resolution_result.get(
        "decision_reason"
    )
)

print(
    "Abstain:",
    resolution_result.get(
        "abstain"
    )
)
