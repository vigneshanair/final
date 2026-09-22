from app.rag.retriever import Retriever
from app.agents.claim_extractor import ClaimExtractionAgent


retriever = Retriever()
claim_agent = ClaimExtractionAgent()

question = "How many days per week can employees work remotely?"

results = retriever.search(
    question,
    n_results=5
)

documents = results.get("documents", [[]])[0]
metadatas = results.get("metadatas", [[]])[0]

print("\nQUESTION:")
print(question)

print("\n" + "=" * 70)
print("RETRIEVED EVIDENCE")
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

    print(f"\nSOURCE: {source}")
    print("-" * 70)
    print(document)

    result = claim_agent.extract(
        document,
        source=source
    )

    print("\nCLAIMS:")

    for claim in result["claims"]:
        print({
            "claim": claim.get("claim"),
            "subject": claim.get("subject"),
            "attribute": claim.get("attribute"),
            "value": claim.get("value"),
            "time_scope": claim.get("time_scope"),
            "condition": claim.get("condition")
        })
