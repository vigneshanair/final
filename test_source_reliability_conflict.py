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
    "How much can employees claim per day "
    "for meal expenses during approved business travel?"
)


files = [
    "data/documents/expense_policy_official.txt",
    "data/documents/expense_policy_unofficial.txt"
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
print("SOURCE RELIABILITY CONFLICT TEST")
print("=" * 70)


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
    "\nPREFERRED SOURCE:",
    source_result.get(
        "preferred_source"
    )
)


print(
    "\nPREFERENCE REASON:",
    source_result.get(
        "preference_reason"
    )
)


print(
    "\nRESOLUTION STRATEGY:",
    resolution_result.get(
        "resolution_strategy"
    )
)


print(
    "\nSELECTED SOURCE:",
    resolution_result.get(
        "selected_source"
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


print("\n" + "=" * 70)
print("EVIDENCE")
print("=" * 70)


for item in evidence:

    print(
        "\nSOURCE:",
        item.get("source")
    )

    print(
        "SOURCE AUTHORITY:",
        item.get("source_authority")
    )

    print(
        "DOCUMENT STATUS:",
        item.get("document_status")
    )

    print(
        "DOCUMENT CONDITION:",
        item.get("document_condition")
    )

    print("CLAIMS:")

    for claim in item.get(
        "claims",
        []
    ):
        print(
            "-",
            claim.get("claim")
        )
