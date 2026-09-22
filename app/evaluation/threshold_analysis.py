import json

from app.security.prompt_detector import PromptInjectionDetector


def load_tests(path="data/test_cases/advanced_attacks.json"):
    with open(path, "r") as file:
        return json.load(file)


def calculate_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn

    accuracy = (
        (tp + tn) / total
        if total
        else 0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    return accuracy, precision, recall, f1


def run():
    detector = PromptInjectionDetector()
    tests = load_tests()

    # Run detector once and save semantic scores
    results = []

    for test in tests:
        result = detector.detect(test["prompt"])

        results.append({
            "id": test["id"],
            "expected_attack": test["should_block"],
            "semantic_similarity": result["semantic_similarity"],
            "regex_attack": len(result["matched_patterns"]) > 0
        })

    thresholds = [
        0.40,
        0.45,
        0.48,
        0.50,
        0.52,
        0.55,
        0.60,
        0.65
    ]

    print("=" * 80)
    print("DEEPSHIELD THRESHOLD ANALYSIS")
    print("=" * 80)

    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for result in results:
            predicted_attack = (
                result["regex_attack"]
                or result["semantic_similarity"] >= threshold
            )

            expected_attack = result["expected_attack"]

            if expected_attack and predicted_attack:
                tp += 1

            elif not expected_attack and predicted_attack:
                fp += 1

            elif expected_attack and not predicted_attack:
                fn += 1

            else:
                tn += 1

        accuracy, precision, recall, f1 = calculate_metrics(
            tp,
            fp,
            fn,
            tn
        )

        print(f"\nThreshold: {threshold:.2f}")
        print(f"Accuracy:  {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1 Score:  {f1:.2f}")
        print(
            f"TP={tp} FP={fp} FN={fn} TN={tn}"
        )

    print("\n" + "=" * 80)
    print("INDIVIDUAL SEMANTIC SCORES")
    print("=" * 80)

    for result in results:
        print(
            result["id"],
            "->",
            result["semantic_similarity"],
            "| Expected attack:",
            result["expected_attack"],
            "| Regex:",
            result["regex_attack"]
        )


if __name__ == "__main__":
    run()
