import json

blank_answer = 0
filled_answer = 0

with open("data/benchmarks/conflicts.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if row.get("conflict_type") == "Complementary information":
            ans = (row.get("correct_answer") or "").strip()
            if ans:
                filled_answer += 1
            else:
                blank_answer += 1

print(f"Complementary info cases with a filled correct_answer (→ conditional_conflict): {filled_answer}")
print(f"Complementary info cases with blank correct_answer (→ ambiguous_question): {blank_answer}")
