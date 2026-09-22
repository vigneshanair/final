from app.agents.claim_extractor import ClaimExtractionAgent
from app.agents.conflict_detector import ConflictDetectionAgent
from app.agents.source_evaluator import SourceEvaluationAgent
from app.agents.resolution_agent import ResolutionAgent
from app.agents.verification_agent import VerificationAgent


claim_agent = ClaimExtractionAgent()
conflict_agent = ConflictDetectionAgent()
source_agent = SourceEvaluationAgent()
resolution_agent = ResolutionAgent()
verification_agent = VerificationAgent()


question = (
    "How many days per week can employees work remotely?"
)


files = [
    "data/documents/employee_policy_2023.txt",
    "data/documents/employee_policy_2025.txt"
]


evidence = []


for path in files:

    with open(path, "r") as file:
        document = file.read()

    source = path.split("/")[-1]

    extracted = claim_agent.extract(
        document,
        source=source
    )

    evidence.append({
        "source": source,
        "publication_year": extracted.get(
            "publication_year"
        ),
        "policy_version": extracted.get(
            "policy_version"
        ),
        "claims": extracted.get(
            "claims",
            []
        )
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
print("TEMPORAL REGRESSION TEST")
print("=" * 70)


print(
    "\nCONFLICT DETECTED:",
    conflict_result.get(
        "conflict_detected"
    )
)


print(
    "CONFLICT TYPE:",
    conflict_result.get(
        "conflict_type"
    )
)


print(
    "RESOLUTION STRATEGY:",
    resolution_result.get(
        "resolution_strategy"
    )
)


print(
    "SELECTED SOURCE:",
    resolution_result.get(
        "selected_source"
    )
)


print(
    "FINAL ANSWER:",
    resolution_result.get(
        "resolved_answer"
    )
)


print(
    "VERIFIED:",
    verification_result.get(
        "verified"
    )
)


print("\nEVIDENCE:")

for item in evidence:

    print("\nSOURCE:", item["source"])
    print("YEAR:", item["publication_year"])
    print("VERSION:", item["policy_version"])

    for claim in item["claims"]:
        print("-", claim.get("claim"))
