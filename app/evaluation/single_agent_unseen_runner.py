import csv
import json
import os
import time

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from app.agents.claim_extractor import ClaimExtractionAgent
from app.rag.single_agent_conflict_rag import SingleAgentConflictRAG


DATASET_PATH = (
    "data/test_cases/conflictguard_eval_unseen.json"
)

RESULTS_DIRECTORY = "results"

CSV_OUTPUT = (
    "results/single_agent_unseen_results.csv"
)

METRICS_OUTPUT = (
    "results/single_agent_unseen_metrics.json"
)


claim_agent = ClaimExtractionAgent()
single_agent = SingleAgentConflictRAG()


def load_dataset():

    with open(
        DATASET_PATH,
        "r"
    ) as file:

        return json.load(file)


def build_evidence(evidence_items):

    evidence = []

    for item in evidence_items:

        source_name = item["source"]
        text = item["text"]

        extracted = claim_agent.extract(
            text,
            source=source_name
        )

        evidence.append({
            "source": source_name,
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

    return evidence


def run_case(test_case):

    question = test_case["question"]

    evidence = build_evidence(
        test_case["evidence"]
    )

    start_time = time.perf_counter()

    result = single_agent.answer(
        question,
        evidence
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    predicted_conflict_type = (
        result.get(
            "conflict_type"
        )
    )

    predicted_strategy = (
        result.get(
            "resolution_strategy"
        )
    )

    predicted_abstain = (
        result.get(
            "abstain",
            False
        )
    )

    expected_conflict_type = (
        test_case[
            "expected_conflict_type"
        ]
    )

    expected_strategy = (
        test_case[
            "expected_resolution_strategy"
        ]
    )

    expected_abstain = (
        test_case[
            "expected_abstain"
        ]
    )

    conflict_type_correct = (
        predicted_conflict_type
        == expected_conflict_type
    )

    strategy_correct = (
        predicted_strategy
        == expected_strategy
    )

    abstain_correct = (
        predicted_abstain
        == expected_abstain
    )

    overall_correct = all([
        conflict_type_correct,
        strategy_correct,
        abstain_correct
    ])

    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "question": question,

        "expected_conflict_type": (
            expected_conflict_type
        ),

        "predicted_conflict_type": (
            predicted_conflict_type
        ),

        "conflict_type_correct": (
            conflict_type_correct
        ),

        "expected_strategy": (
            expected_strategy
        ),

        "predicted_strategy": (
            predicted_strategy
        ),

        "strategy_correct": (
            strategy_correct
        ),

        "expected_abstain": (
            expected_abstain
        ),

        "predicted_abstain": (
            predicted_abstain
        ),

        "abstain_correct": (
            abstain_correct
        ),

        "overall_correct": (
            overall_correct
        ),

        "latency_seconds": round(
            elapsed,
            4
        ),

        "final_answer": (
            result.get(
                "answer",
                ""
            )
        ),

        "explanation": (
            result.get(
                "explanation",
                ""
            )
        )
    }


def save_csv(results):

    if not results:
        return

    os.makedirs(
        RESULTS_DIRECTORY,
        exist_ok=True
    )

    with open(
        CSV_OUTPUT,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(results)


def calculate_metrics(results):

    y_true = [
        item[
            "expected_conflict_type"
        ]
        for item in results
    ]

    y_pred = [
        item[
            "predicted_conflict_type"
        ]
        for item in results
    ]

    conflict_accuracy = (
        accuracy_score(
            y_true,
            y_pred
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    strategy_accuracy = (
        sum(
            item["strategy_correct"]
            for item in results
        )
        / len(results)
    )

    abstention_accuracy = (
        sum(
            item["abstain_correct"]
            for item in results
        )
        / len(results)
    )

    overall_accuracy = (
        sum(
            item["overall_correct"]
            for item in results
        )
        / len(results)
    )

    average_latency = (
        sum(
            item["latency_seconds"]
            for item in results
        )
        / len(results)
    )

    per_category = {}

    categories = sorted(
        set(
            item["category"]
            for item in results
        )
    )

    for category in categories:

        category_results = [
            item
            for item in results
            if item["category"]
            == category
        ]

        type_correct = sum(
            item["conflict_type_correct"]
            for item in category_results
        )

        strategy_correct = sum(
            item["strategy_correct"]
            for item in category_results
        )

        overall_correct = sum(
            item["overall_correct"]
            for item in category_results
        )

        per_category[category] = {
            "total": len(
                category_results
            ),

            "conflict_type_accuracy": (
                type_correct
                / len(category_results)
            ),

            "strategy_accuracy": (
                strategy_correct
                / len(category_results)
            ),

            "overall_accuracy": (
                overall_correct
                / len(category_results)
            )
        }

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return {
        "total_cases": len(results),

        "conflict_type_accuracy": round(
            conflict_accuracy,
            4
        ),

        "macro_precision": round(
            macro_precision,
            4
        ),

        "macro_recall": round(
            macro_recall,
            4
        ),

        "macro_f1": round(
            macro_f1,
            4
        ),

        "resolution_strategy_accuracy": round(
            strategy_accuracy,
            4
        ),

        "abstention_accuracy": round(
            abstention_accuracy,
            4
        ),

        "overall_pipeline_accuracy": round(
            overall_accuracy,
            4
        ),

        "average_latency_seconds": round(
            average_latency,
            4
        ),

        "per_category": (
            per_category
        ),

        "classification_report": (
            report
        )
    }


def save_metrics(metrics):

    os.makedirs(
        RESULTS_DIRECTORY,
        exist_ok=True
    )

    with open(
        METRICS_OUTPUT,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )


def run():

    dataset = load_dataset()

    results = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SINGLE-AGENT UNSEEN EVALUATION"
    )

    print(
        "=" * 70
    )

    for index, test_case in enumerate(
        dataset,
        start=1
    ):

        print(
            f"\n[{index}/{len(dataset)}] "
            f"{test_case['id']} "
            f"({test_case['category']})"
        )

        try:

            result = run_case(
                test_case
            )

            results.append(
                result
            )

            print(
                "Expected:",
                result[
                    "expected_conflict_type"
                ]
            )

            print(
                "Predicted:",
                result[
                    "predicted_conflict_type"
                ]
            )

            print(
                "Strategy:",
                result[
                    "predicted_strategy"
                ]
            )

            print(
                "Abstain:",
                result[
                    "predicted_abstain"
                ]
            )

            print(
                "Latency:",
                result[
                    "latency_seconds"
                ],
                "seconds"
            )

            print(
                "Result:",
                (
                    "PASS"
                    if result[
                        "overall_correct"
                    ]
                    else "FAIL"
                )
            )

        except Exception as error:

            print(
                "ERROR:",
                error
            )

    if not results:
        return

    save_csv(results)

    metrics = calculate_metrics(
        results
    )

    save_metrics(metrics)

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SINGLE-AGENT EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Total Cases:",
        metrics[
            "total_cases"
        ]
    )

    print(
        "Conflict Type Accuracy:",
        f"{metrics['conflict_type_accuracy'] * 100:.2f}%"
    )

    print(
        "Macro Precision:",
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        "Macro Recall:",
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        "Macro F1:",
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        "Resolution Strategy Accuracy:",
        f"{metrics['resolution_strategy_accuracy'] * 100:.2f}%"
    )

    print(
        "Abstention Accuracy:",
        f"{metrics['abstention_accuracy'] * 100:.2f}%"
    )

    print(
        "Overall Pipeline Accuracy:",
        f"{metrics['overall_pipeline_accuracy'] * 100:.2f}%"
    )

    print(
        "Average Latency:",
        f"{metrics['average_latency_seconds']:.4f} seconds"
    )

    print(
        "\nPER-CATEGORY RESULTS"
    )

    print(
        "-" * 70
    )

    for category, data in metrics[
        "per_category"
    ].items():

        print(category)

        print(
            "  Type Accuracy:",
            f"{data['conflict_type_accuracy'] * 100:.2f}%"
        )

        print(
            "  Strategy Accuracy:",
            f"{data['strategy_accuracy'] * 100:.2f}%"
        )

        print(
            "  Overall Accuracy:",
            f"{data['overall_accuracy'] * 100:.2f}%"
        )

    print(
        "\nResults saved to:"
    )

    print(CSV_OUTPUT)
    print(METRICS_OUTPUT)


if __name__ == "__main__":
    run()
