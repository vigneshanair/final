from app.agents.coordinator import ConflictGuardCoordinator


coordinator = ConflictGuardCoordinator()

question = "How many paid vacation days do employees receive per year?"

result = coordinator.answer(
    question,
    n_results=2
)

print("\n" + "=" * 70)
print("DIRECT CONFLICT TEST")
print("=" * 70)

print("\nQUESTION:")
print(result.get("question"))

print("\nFINAL ANSWER:")
print(result.get("final_answer"))

print("\nCONFLICT DETECTED:")
print(result.get("conflict_detected"))

print("\nCONFLICT TYPE:")
print(result.get("conflict_type"))

print("\nRESOLUTION STRATEGY:")
print(result.get("resolution_strategy"))

print("\nSELECTED SOURCE:")
print(result.get("selected_source"))

print("\nVERIFIED:")
print(result.get("verified"))

print("\nABSTAINED:")
print(result.get("abstained"))

print("\nCONFLICT EXPLANATION:")
print(result.get("conflict_explanation"))

print("\nEVIDENCE:")

for item in result.get("evidence", []):
    print("\nSOURCE:", item.get("source"))

    for claim in item.get("claims", []):
        print("-", claim.get("claim"))
