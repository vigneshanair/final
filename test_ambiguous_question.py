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
    "data/documents/remote_work_full_time.txt",
    "data/documents/remote_work_part_time.txt"
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
        "source_authority": extracted.get(
            "source_authority"
        ),
        "document_status": extracted.get(
            "document_status"
        ),
        "document_condition": extracted.get(
            "document_condition"
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
print("AMBIGUOUS QUESTION TEST")
print("=" * 70)


print(
    "\nQUESTION:",
    question
)


print(
    "\nCONFLICT DETECTED:",
    conflict_result.get(
        "conflict_detected"
    )
)


print(
    "\nCONFLICT TYPE:",
    conflict_result.get(
        "conflict_type"
    )
)


print(
    "\nCONFLICT EXPLANATION:",
    conflict_result.get(
        "explanation"
    )
)


print(
    "\nRESOLUTION STRATEGY:",
    resolution_result.get(
        "resolution_strategy"
    )
)


print(
    "\nFINAL ANSWER:",
    resolution_result.get(
        "resolved_answer"
    )
)


print(
    "\nVERIFIED:",
    verification_result.get(
        "verified"
    )
)


print(
    "\nABSTAINED:",
    resolution_result.get(
        "abstain"
    )
)


print("\nEVIDENCE:")

for item in evidence:

    print("\nSOURCE:", item["source"])

    print(
        "DOCUMENT CONDITION:",
        item.get(
            "document_condition"
        )
    )

    for claim in item.get(
        "claims",
        []
    ):
        print(
            "-",
            claim.get("claim"),
            "| Condition:",
            claim.get("condition")
        )
