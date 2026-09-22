import subprocess


tests = [
    {
        "name": "Temporal Conflict",
        "file": "test_temporal_regression.py",
        "expected": [
            "CONFLICT TYPE: temporal_conflict",
            "RESOLUTION STRATEGY: prefer_newer_authoritative_source",
            "VERIFIED: True"
        ]
    },
    {
        "name": "Direct Factual Conflict",
        "file": "test_direct_conflict.py",
        "expected": [
            "CONFLICT TYPE:",
            "direct_factual_conflict",
            "RESOLUTION STRATEGY:",
            "present_both_positions",
            "VERIFIED:",
            "True"
        ]
    },
    {
        "name": "Conditional Conflict",
        "file": "test_conditional_conflict.py",
        "expected": [
            "CONFLICT TYPE:",
            "conditional_conflict",
            "RESOLUTION STRATEGY:",
            "explain_conditional_difference",
            "VERIFIED:",
            "True"
        ]
    },
    {
        "name": "Source Reliability Conflict",
        "file": "test_source_reliability_conflict.py",
        "expected": [
            "CONFLICT TYPE: source_reliability_conflict",
            "RESOLUTION STRATEGY: prefer_authoritative_source",
            "VERIFIED: True"
        ]
    },
    {
        "name": "Ambiguous Question",
        "file": "test_ambiguous_question.py",
        "expected": [
            "CONFLICT TYPE: ambiguous_question",
            "RESOLUTION STRATEGY: ask_for_clarification",
            "VERIFIED: True"
        ]
    },
    {
        "name": "Insufficient Evidence",
        "file": "test_insufficient_evidence.py",
        "expected": [
            "CONFLICT TYPE: insufficient_evidence",
            "RESOLUTION STRATEGY: abstain",
            "VERIFIED: True",
            "ABSTAINED: True"
        ]
    },
    {
        "name": "No Conflict",
        "file": "test_no_conflict.py",
        "expected": [
            "CONFLICT TYPE: no_conflict",
            "RESOLUTION STRATEGY: no_resolution_needed",
            "FINAL ANSWER: Manager approval is required for remote work.",
            "VERIFIED: True"
        ]
    }
]


passed = 0
failed = 0


print("\n" + "=" * 70)
print("CONFLICTGUARD FULL REGRESSION SUITE")
print("=" * 70)


for test in tests:

    print(f"\nRunning: {test['name']}")
    print("-" * 70)

    result = subprocess.run(
        ["python", test["file"]],
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr

    if result.returncode != 0:
        print("RESULT: FAIL")
        print("Reason: Test crashed.")
        print(output)

        failed += 1
        continue

    missing = []

    for expected_text in test["expected"]:

        if expected_text not in output:
            missing.append(expected_text)

    if not missing:

        print("RESULT: PASS")
        passed += 1

    else:

        print("RESULT: FAIL")
        print("Missing expected output:")

        for item in missing:
            print("-", item)

        print("\nFULL OUTPUT:")
        print(output)

        failed += 1


print("\n" + "=" * 70)
print("REGRESSION SUMMARY")
print("=" * 70)

print("TOTAL:", len(tests))
print("PASSED:", passed)
print("FAILED:", failed)

if failed == 0:
    print("\nALL CONFLICTGUARD CORE TESTS PASSED.")
else:
    print(
        "\nSome tests failed. "
        "Review the outputs above."
    )
