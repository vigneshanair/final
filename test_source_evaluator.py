from app.rag.retriever import Retriever
from app.agents.claim_extractor import ClaimExtractionAgent
from app.agents.conflict_detector import ConflictDetectionAgent
from app.agents.source_evaluator import SourceEvaluationAgent


retriever = Retriever()

claim_agent = ClaimExtractionAgent()

conflict_agent = ConflictDetectionAgent()

source_agent = SourceEvaluationAgent()


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


print("\n" + "=" * 70)

print("SOURCE EVALUATION")

print("=" * 70)


for item in source_result.get(
    "source_evaluations",
    []
):

    print(
        "\nSOURCE:",
        item.get("source")
    )

    print(
        "Relevance:",
        item.get("relevance")
    )

    print(
        "Publication Year:",
        item.get("publication_year")
    )

    print(
        "Policy Version:",
        item.get("policy_version")
    )

    print(
        "Supersedes Another Source:",
        item.get(
            "supersedes_other_source"
        )
    )

    print(
        "Superseded Source:",
        item.get(
            "superseded_source"
        )
    )

    print(
        "Applicability:",
        item.get("applicability")
    )

    print(
        "Reason:",
        item.get(
            "evaluation_reason"
        )
    )


print("\n" + "=" * 70)

print("PREFERRED SOURCE")

print("=" * 70)

print(
    source_result.get(
        "preferred_source"
    )
)


print("\nPREFERENCE REASON:")

print(
    source_result.get(
        "preference_reason"
    )
)
