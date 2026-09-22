"""Builds data/test_cases/external_conflicts_sample.json, a stratified
sample from the google-research-datasets/rag_conflicts benchmark, used for
ConflictGuard's external evaluation (see app/evaluation/external_conflicts_runner.py).

Requires data/benchmarks/conflicts.jsonl locally (not tracked in git --
fetch it first):

    curl -L -o data/benchmarks/conflicts.jsonl \\
        https://raw.githubusercontent.com/google-research-datasets/rag_conflicts/refs/heads/main/conflicts.jsonl

Re-run this script any time to regenerate the sample with a different seed
or sample size; the checked-in output is what the runner actually reads.
"""

import json
import random

SOURCE_PATH = "data/benchmarks/conflicts.jsonl"
OUTPUT_PATH = "data/test_cases/external_conflicts_sample.json"

MAX_PASSAGES = 3
MAX_CHARS = 800

# Sample sizes per (original label, correct_answer filled?) bucket. The
# "Complementary information" split follows check_complementary_split.py:
# a filled correct_answer means the sources combine into one answer under
# a condition (-> conditional_conflict); a blank correct_answer means no
# single answer is correct without more context (-> ambiguous_question).
# "Conflict due to misinformation" has no equivalent in ConflictGuard's
# taxonomy (it's about factual truth, not conflict structure) and is kept
# unscored, for qualitative review only.
SAMPLE_PLAN = [
    # (original_conflict_type, requires_filled_answer, mapped_type, count)
    ("No conflict", None, "no_conflict", 12),
    ("Conflict due to outdated information", None, "temporal_conflict", 12),
    ("Conflicting opinions and research outcomes", None, "direct_factual_conflict", 12),
    ("Complementary information", False, "ambiguous_question", 12),
    ("Complementary information", True, "conditional_conflict", 7),
    ("Conflict due to misinformation", None, None, 5),
]

random.seed(7)


def load_records():
    with open(SOURCE_PATH) as file:
        return [json.loads(line) for line in file]


def passage_text(search_result):
    text = search_result.get("short_text") or search_result.get("response_str") or ""
    text = text.strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0] + "..."
    return text


def passage_source_line(search_result):
    date = search_result.get("date")
    if date and date != "NA" and len(date) >= 4 and date[:4].isdigit():
        return f"Publication Year: {date[:4]}\n"
    return ""


def build_evidence(record):
    evidence = []
    for search_result in record["search_results"][:MAX_PASSAGES]:
        title = search_result.get("title") or search_result.get("url") or "untitled source"
        text = passage_text(search_result)
        if not text:
            continue
        evidence.append({
            "source": title,
            "text": passage_source_line(search_result) + text,
            "url": search_result.get("url"),
        })
    return evidence


def matches_bucket(record, requires_filled_answer):
    if requires_filled_answer is None:
        return True
    has_answer = bool((record.get("correct_answer") or "").strip())
    return has_answer == requires_filled_answer


def main():
    records = load_records()

    by_label = {}
    for record in records:
        by_label.setdefault(record["conflict_type"], []).append(record)

    cases = []
    case_index = 0

    for original_label, requires_filled_answer, mapped_type, count in SAMPLE_PLAN:
        pool = [
            record for record in by_label.get(original_label, [])
            if matches_bucket(record, requires_filled_answer)
        ]
        random.shuffle(pool)

        for record in pool[:count]:
            evidence = build_evidence(record)
            if not evidence:
                continue

            case_index += 1
            cases.append({
                "id": f"EXT{case_index:03d}",
                "dataset_source": record.get("source"),
                "original_conflict_type": original_label,
                "expected_conflict_type": mapped_type,
                "scored": mapped_type is not None,
                "question": record["question"],
                "reference_answer": record.get("correct_answer"),
                "evidence": evidence,
            })

    random.shuffle(cases)
    for index, case in enumerate(cases, start=1):
        case["id"] = f"EXT{index:03d}"

    with open(OUTPUT_PATH, "w") as file:
        json.dump(cases, file, indent=2)

    print(f"Wrote {len(cases)} cases to {OUTPUT_PATH}")

    counts = {}
    for case in cases:
        key = (case["original_conflict_type"], case["expected_conflict_type"])
        counts[key] = counts.get(key, 0) + 1
    for key, value in sorted(counts.items()):
        print(f"  {key[0]:45s} -> {str(key[1]):25s} {value}")


if __name__ == "__main__":
    main()
