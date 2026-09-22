import csv
import json
import os

from app.security.prompt_detector import PromptInjectionDetector


TEST_PATH = "data/test_cases/heldout_prompts.json"
RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "heldout_results.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "heldout_metrics.json")


def load_tests():
    with open(TEST_PATH, "r") as file:
        return json.load(file)


def run():
    # Frozen detector -- do not modify threshold or attack examples
    detector = PromptInjectionDetector()
    tests = load_tests()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    rows = []

    for test in tests:
        result = detector.detect(test["prompt"])

        expected = test["should_block"]
        predicted = result["is_attack"]

        if expected and predicted:
            classification = "TP"
            tp += 1

        elif not expected and not predicted:
            classification = "TN"
            tn += 1

        elif not expected and predicted:
            classification = "FP"
            fp += 1

        else:
            classification = "FN"
            fn += 1

        rows.append({
            "id": test["id"],
            "prompt": test["prompt"],
            "expected_attack": expected,
            "predicted_attack": predicted,
            "classification": classification,
            "correct": expected == predicted,
            "semantic_similarity": result["semantic_similarity"],
            "semantic_attack": result["semantic_attack"],
            "regex_match": len(result["matched_patterns"]) > 0,
            "matched_patterns": " | ".join(result["matched_patterns"])
        })

        print(
            f'{test["id"]}: '
            f'Expected={expected} '
            f'Predicted={predicted} '
            f'Similarity={result["semantic_similarity"]:.4f} '
            f'[{classification}]'
        )

    total = len(tests)

    accuracy = (tp + tn) / total if total else 0

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics = {
        "total_prompts": total,
        "attack_prompts": sum(
            test["should_block"] for test in tests
        ),
        "benign_prompts": sum(
            not test["should_block"] for test in tests
        ),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "semantic_threshold": detector.semantic_threshold
    }

    # Save detailed CSV
    with open(CSV_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    # Save summary metrics
    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "=" * 60)
    print("DEEPSHIELD HELD-OUT EVALUATION")
    print("=" * 60)

    print(f"Total Prompts:    {total}")
    print(f"Threshold:        {detector.semantic_threshold}")
    print()

    print(f"True Positives:   {tp}")
    print(f"True Negatives:   {tn}")
    print(f"False Positives:  {fp}")
    print(f"False Negatives:  {fn}")

    print()
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nMisclassified prompts:")

    failures = [
        row for row in rows
        if not row["correct"]
    ]

    if not failures:
        print("None")
    else:
        for failure in failures:
            print(
                f'{failure["id"]} [{failure["classification"]}] '
                f'Similarity={failure["semantic_similarity"]}'
            )
            print(f'  {failure["prompt"]}')

    print("\nResults saved to:")
    print(CSV_PATH)
    print(METRICS_PATH)


if __name__ == "__main__":
    run()
