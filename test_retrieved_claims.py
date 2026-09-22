from app.rag.retriever import Retriever
from app.agents.claim_extractor import ClaimExtractionAgent


retriever = Retriever()
claim_agent = ClaimExtractionAgent()

question = "What are the employee work policies?"

results = retriever.search(
    question,
    n_results=3
)

documents = results.get("documents", [[]])[0]
metadatas = results.get("metadatas", [[]])[0]

print("\nQUESTION:")
print(question)

print("\n" + "=" * 70)
print("RETRIEVED DOCUMENTS AND CLAIMS")
print("=" * 70)

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

    print(f"\nSOURCE {index + 1}: {source}")
    print("-" * 70)

    print("\nPASSAGE:")
    print(document)

    result = claim_agent.extract(
        document,
        source=source
    )

    print("\nEXTRACTED CLAIMS:")

    if not result["claims"]:
        print("No claims extracted.")
        continue

    for claim_index, claim in enumerate(
        result["claims"],
        start=1
    ):
        print(f"\nClaim {claim_index}")
        print("Claim:", claim.get("claim"))
        print("Subject:", claim.get("subject"))
        print("Attribute:", claim.get("attribute"))
        print("Value:", claim.get("value"))
        print("Time Scope:", claim.get("time_scope"))
        print("Condition:", claim.get("condition"))
