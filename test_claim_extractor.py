from app.agents.claim_extractor import ClaimExtractionAgent


agent = ClaimExtractionAgent()

passage = """
According to the 2025 employee policy, employees may work remotely
two days per week. Employees must receive manager approval before
working remotely.
"""

result = agent.extract(
    passage,
    source="employee_policy_2025.txt"
)

print("\nSOURCE:")
print(result["source"])

print("\nEXTRACTED CLAIMS:")

for index, claim in enumerate(result["claims"], start=1):
    print(f"\nClaim {index}")
    print("Claim:", claim.get("claim"))
    print("Subject:", claim.get("subject"))
    print("Attribute:", claim.get("attribute"))
    print("Value:", claim.get("value"))
    print("Time Scope:", claim.get("time_scope"))
    print("Condition:", claim.get("condition"))
