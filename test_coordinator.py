from app.agents.coordinator import ConflictGuardCoordinator


coordinator = ConflictGuardCoordinator()


question = (
    "How many days per week can employees work remotely?"
)


result = coordinator.answer(
    question,
    n_results=2
)


print("\n" + "=" * 70)
print("CONFLICTGUARD FINAL RESULT")
print("=" * 70)


print(
    "\nQUESTION:",
    result.get("question")
)


print(
    "\nFINAL ANSWER:",
    result.get("final_answer")
)


print(
    "\nCONFLICT DETECTED:",
    result.get("conflict_detected")
)


print(
    "CONFLICT TYPE:",
    result.get("conflict_type")
)


print(
    "RESOLUTION STRATEGY:",
    result.get("resolution_strategy")
)


print(
    "SELECTED SOURCE:",
    result.get("selected_source")
)


print(
    "VERIFIED:",
    result.get("verified")
)


print(
    "ABSTAINED:",
    result.get("abstained")
)


print("\nCONFLICT EXPLANATION:")

print(
    result.get("conflict_explanation")
)


print("\n" + "=" * 70)
print("RETRIEVED EVIDENCE")
print("=" * 70)


for item in result.get(
    "evidence",
    []
):

    print(
        "\nSOURCE:",
        item.get("source")
    )

    print(
        "YEAR:",
        item.get("publication_year")
    )

    print(
        "VERSION:",
        item.get("policy_version")
    )

    for claim in item.get(
        "claims",
        []
    ):

        print(
            "-",
            claim.get("claim")
        )
