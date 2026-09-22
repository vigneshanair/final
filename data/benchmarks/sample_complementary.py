import json

count = 0
with open("data/benchmarks/conflicts.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if row.get("conflict_type") == "Complementary information":
            count += 1
            print(f"\n{'='*80}")
            print(f"CASE #{count}")
            print(f"QUESTION: {row.get('question')}")
            print(f"CORRECT ANSWER: {row.get('correct_answer', 'N/A')}")
            print(f"\nSEARCH RESULTS ({len(row.get('search_results', []))} total):")
            for i, sr in enumerate(row.get("search_results", [])[:3]):
                print(f"\n  [{i+1}] {sr.get('title')}")
                print(f"      snippet: {sr.get('short_text', sr.get('snippet',''))[:300]}")
            if count >= 8:
                break
