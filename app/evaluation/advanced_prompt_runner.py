import json

from app.security.prompt_detector import PromptInjectionDetector


def load_tests(path="data/test_cases/advanced_attacks.json"):
    with open(path, "r") as file:
        return json.load(file)


def run():
    detector = PromptInjectionDetector()
    tests = load_tests()

    correct = 0
    total = len(tests)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    failures = []

    for test in tests:
        result = detector.detect(test["prompt"])

        predicted_attack = result["is_attack"]
        expected_attack = test["should_block"]

        if predicted_attack == expected_attack:
            correct += 1

        if expected_attack and predicted_attack:
            true_positives += 1

        elif not expected_attack and predicted_attack:
            false_positives += 1

            failures.append({
                "id": test["id"],
                "type": "FALSE POSITIVE",
                "prompt": test["prompt"],
                "similarity": result["semantic_similarity"]
            })

        elif expected_attack and not predicted_attack:
            false_negatives += 1

            failures.append({
                "id": test["id"],
                "type": "FALSE NEGATIVE",
                "prompt": test["prompt"],
                "similarity": result["semantic_similarity"]
            })

        else:
            true_negatives += 1

        print("=" * 60)
        print("TEST:", test["id"])
        print("PROMPT:", test["prompt"])
        print("EXPECTED ATTACK:", expected_attack)
        print("PREDICTED ATTACK:", predicted_attack)
        print("SEMANTIC SIMILARITY:", result["semantic_similarity"])
        print("SEMANTIC ATTACK:", result["semantic_attack"])
        print("MATCHED:", result["matched_patterns"])

    accuracy = correct / total * 100 if total else 0

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    print("\n" + "=" * 60)
    print("ADVANCED PROMPT DETECTOR METRICS")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    print("True Positives:", true_positives)
    print("False Positives:", false_positives)
    print("False Negatives:", false_negatives)
    print("True Negatives:", true_negatives)

    print("\n" + "=" * 60)
    print("MISCLASSIFIED TESTS")
    print("=" * 60)

    if not failures:
        print("None")
    else:
        for failure in failures:
            print("ID:", failure["id"])
            print("TYPE:", failure["type"])
            print("PROMPT:", failure["prompt"])
            print("SIMILARITY:", failure["similarity"])
            print("-" * 60)


if __name__ == "__main__":
    run()
