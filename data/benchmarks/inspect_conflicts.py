import json
from collections import Counter

conflict_types = Counter()
sources = Counter()
total = 0

with open("data/benchmarks/conflicts.jsonl") as f:
    for line in f:
        row = json.loads(line)
        conflict_types[row.get("conflict_type", "MISSING")] += 1
        sources[row.get("source", "unknown")] += 1
        total += 1

print(f"Total instances: {total}\n")

print("=== Conflict Type Distribution ===")
for k, v in conflict_types.most_common():
    print(f"{v:4d}  {k}")

print("\n=== Source Dataset Distribution ===")
for k, v in sources.most_common():
    print(f"{v:4d}  {k}")
