from app.rag.single_agent_conflict_rag import SingleAgentConflictRAG


agent = SingleAgentConflictRAG()


question = (
    "How many days per week can employees "
    "work remotely?"
)


evidence = [
    {
        "source": "employee_policy_2023.txt",
        "publication_year": "2023",
        "policy_version": "1.0",
        "claims": [
            {
                "claim": (
                    "Employees may work remotely "
                    "three days per week."
                )
            }
        ]
    },
    {
        "source": "employee_policy_2025.txt",
        "publication_year": "2025",
        "policy_version": "2.0",
        "claims": [
            {
                "claim": (
                    "Employees may work remotely "
                    "two days per week."
                )
            },
            {
                "claim": (
                    "This policy supersedes the "
                    "2023 remote work policy."
                )
            }
        ]
    }
]


result = agent.answer(
    question,
    evidence
)


print("\n" + "=" * 70)
print("SINGLE-AGENT CONFLICT-AWARE RAG TEST")
print("=" * 70)

print(
    "\nCONFLICT TYPE:",
    result.get("conflict_type")
)

print(
    "\nRESOLUTION STRATEGY:",
    result.get(
        "resolution_strategy"
    )
)

print(
    "\nSELECTED SOURCE:",
    result.get(
        "selected_source"
    )
)

print(
    "\nANSWER:",
    result.get("answer")
)

print(
    "\nABSTAINED:",
    result.get("abstain")
)

print(
    "\nEXPLANATION:",
    result.get("explanation")
)
