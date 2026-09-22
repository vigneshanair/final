"""External evaluation of the ConflictGuard multi-agent pipeline against a
sample of the google-research-datasets/rag_conflicts benchmark
(data/test_cases/external_conflicts_sample.json, built by
data/benchmarks/build_external_sample.py).

Unlike the internal benchmark, this dataset's conflict-type labels only
map cleanly onto three of ConflictGuard's categories directly, plus a
two-way split of its "Complementary information" label (see
build_external_sample.py for the mapping and its rationale). Cases where
no defensible mapping exists ("Conflict due to misinformation") are still
run for qualitative review but excluded from the accuracy metric.

There is no ground truth for resolution strategy or abstention in this
dataset, so -- unlike conflictguard_unseen_runner.py -- this runner only
scores conflict-type classification and reports resolution/abstention/
verification as descriptive statistics.
"""

import csv
import json
import os
import time

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from app.agents.claim_extractor import ClaimExtractionAgent
from app.agents.conflict_detector import ConflictDetectionAgent
from app.agents.source_evaluator import SourceEvaluationAgent
from app.agents.resolution_agent import ResolutionAgent
from app.agents.verification_agent import VerificationAgent


DATASET_PATH = "data/test_cases/external_conflicts_sample.json"
RESULTS_DIRECTORY = "results"
CSV_OUTPUT = "results/external_conflicts_results.csv"
METRICS_OUTPUT = "results/external_conflicts_metrics.json"


claim_agent = ClaimExtractionAgent()
conflict_agent = ConflictDetectionAgent()
source_agent = SourceEvaluationAgent()
resolution_agent = ResolutionAgent()
verification_agent = VerificationAgent()


def load_dataset():
    with open(DATASET_PATH, "r") as file:
        return json.load(file)


def build_evidence(evidence_items):
    evidence = []

    for item in evidence_items:
        extracted = claim_agent.extract(item["text"], source=item["source"])

        evidence.append({
            "source": item["source"],
            "publication_year": extracted.get("publication_year"),
            "policy_version": extracted.get("policy_version"),
            "source_authority": extracted.get("source_authority"),
            "document_status": extracted.get("document_status"),
            "document_condition": extracted.get("document_condition"),
            "claims": extracted.get("claims", []),
        })

    return evidence


def run_case(test_case):
    question = test_case["question"]
    evidence = build_evidence(test_case["evidence"])

    start_time = time.perf_counter()

    conflict_result = conflict_agent.detect(question, evidence)
    source_result = source_agent.evaluate(question, evidence, conflict_result)
    resolution_result = resolution_agent.resolve(question, evidence, conflict_result, source_result)
    verification_result = verification_agent.verify(question, evidence, resolution_result)

    elapsed = time.perf_counter() - start_time

    predicted_conflict_type = conflict_result.get("conflict_type")
    expected_conflict_type = test_case["expected_conflict_type"]

    return {
        "id": test_case["id"],
        "dataset_source": test_case["dataset_source"],
        "original_conflict_type": test_case["original_conflict_type"],
        "scored": test_case["scored"],
        "question": question,

        "expected_conflict_type": expected_conflict_type,
        "predicted_conflict_type": predicted_conflict_type,
        "conflict_type_correct": (
            predicted_conflict_type == expected_conflict_type
            if test_case["scored"]
            else None
        ),

        "resolution_strategy": resolution_result.get("resolution_strategy"),
        "selected_source": resolution_result.get("selected_source"),
        "abstained": resolution_result.get("abstain", False),
        "verified": verification_result.get("verified", False),

        "latency_seconds": round(elapsed, 4),

        "reference_answer": test_case.get("reference_answer", ""),
        "final_answer": resolution_result.get("resolved_answer", ""),
        "conflict_explanation": conflict_result.get("explanation", ""),
    }


def save_csv(results):
    if not results:
        return

    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    with open(CSV_OUTPUT, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def calculate_metrics(results):
    scored = [item for item in results if item["scored"]]
    unscored = [item for item in results if not item["scored"]]

    y_true = [item["expected_conflict_type"] for item in scored]
    y_pred = [item["predicted_conflict_type"] for item in scored]

    conflict_accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    per_category = {}
    for category in sorted(set(item["original_conflict_type"] for item in scored)):
        category_results = [item for item in scored if item["original_conflict_type"] == category]
        correct = sum(item["conflict_type_correct"] for item in category_results)
        per_category[category] = {
            "total": len(category_results),
            "conflict_type_accuracy": round(correct / len(category_results), 4),
        }

    unscored_distribution = {}
    for item in unscored:
        label = item["predicted_conflict_type"]
        unscored_distribution[label] = unscored_distribution.get(label, 0) + 1

    abstain_rate = sum(item["abstained"] for item in results) / len(results)
    verified_rate = sum(item["verified"] for item in results) / len(results)
    average_latency = sum(item["latency_seconds"] for item in results) / len(results)

    return {
        "total_cases": len(results),
        "scored_cases": len(scored),
        "unscored_cases": len(unscored),

        "conflict_type_accuracy": round(conflict_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),

        "abstain_rate": round(abstain_rate, 4),
        "verified_rate": round(verified_rate, 4),
        "average_latency_seconds": round(average_latency, 4),

        "per_category": per_category,
        "classification_report": report,

        "unscored_note": (
            "\"Conflict due to misinformation\" cases have no equivalent in "
            "ConflictGuard's conflict-structure taxonomy (they concern factual "
            "truth, not evidence disagreement) and are excluded from the "
            "accuracy metrics above. Their predicted conflict-type distribution "
            "is reported here for qualitative review only."
        ),
        "unscored_predicted_distribution": unscored_distribution,
    }


def save_metrics(metrics):
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    with open(METRICS_OUTPUT, "w") as file:
        json.dump(metrics, file, indent=2)


def run():
    dataset = load_dataset()
    results = []

    print("\n" + "=" * 70)
    print("EXTERNAL EVALUATION (rag_conflicts benchmark)")
    print("=" * 70)

    for index, test_case in enumerate(dataset, start=1):
        print(f"\n[{index}/{len(dataset)}] {test_case['id']} ({test_case['original_conflict_type']})")

        try:
            result = run_case(test_case)
            results.append(result)

            print("Expected:", result["expected_conflict_type"])
            print("Predicted:", result["predicted_conflict_type"])
            print("Latency:", result["latency_seconds"], "seconds")

            if result["scored"]:
                print("Result:", "PASS" if result["conflict_type_correct"] else "FAIL")
            else:
                print("Result: (unscored)")

        except Exception as error:
            print("ERROR:", error)

    if not results:
        return

    save_csv(results)
    metrics = calculate_metrics(results)
    save_metrics(metrics)

    print("\n" + "=" * 70)
    print("EXTERNAL EVALUATION SUMMARY")
    print("=" * 70)
    print("Total cases:", metrics["total_cases"], f"({metrics['scored_cases']} scored)")
    print("Conflict Type Accuracy:", f"{metrics['conflict_type_accuracy'] * 100:.2f}%")
    print("Macro F1:", f"{metrics['macro_f1']:.4f}")
    print("Abstain Rate:", f"{metrics['abstain_rate'] * 100:.2f}%")
    print("Verified Rate:", f"{metrics['verified_rate'] * 100:.2f}%")
    print("Average Latency:", f"{metrics['average_latency_seconds']:.4f} seconds")

    print("\nPER-CATEGORY RESULTS")
    print("-" * 70)
    for category, data in metrics["per_category"].items():
        print(f"{category}: {data['conflict_type_accuracy'] * 100:.2f}% ({data['total']} cases)")

    print("\nResults saved to:")
    print(CSV_OUTPUT)
    print(METRICS_OUTPUT)


if __name__ == "__main__":
    run()
