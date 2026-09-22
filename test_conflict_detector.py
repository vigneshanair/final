from app.rag.retriever import Retriever
from app.agents.claim_extractor import ClaimExtractionAgent
from app.agents.conflict_detector import ConflictDetectionAgent


retriever = Retriever()
claim_agent = ClaimExtractionAgent()
conflict_agent = ConflictDetectionAgent()


question = "How many days per week can employees work remotely?"


results = retriever.search(
    question,
    n_results=2
)


documents = results.get("documents", [[]])[0]
metadatas = results.get("metadatas", [[]])[0]


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
        "publication_year": extracted["publication_year"],
        "policy_version": extracted["policy_version"],
        "claims": extracted["claims"]
    })


print("\nQUESTION:")
print(question)


print("\nEVIDENCE:")

for item in evidence:

    print("\nSOURCE:", item["source"])

    for claim in item["claims"]:
        print(
            "-",
            claim["claim"],
            "| Year:",
            claim["time_scope"]
        )


result = conflict_agent.detect(
    question,
    evidence
)


print("\n" + "=" * 70)
print("CONFLICT DETECTION RESULT")
print("=" * 70)

print(
    "Conflict Detected:",
    result.get("conflict_detected")
)

print(
    "Conflict Type:",
    result.get("conflict_type")
)

print(
    "Confidence:",
    result.get("confidence")
)

print(
    "Explanation:",
    result.get("explanation")
)


print("\nCONFLICTING CLAIMS:")

for claim in result.get(
    "conflicting_claims",
    []
):

    print(
        claim.get("source"),
        "→",
        claim.get("claim")
    )
