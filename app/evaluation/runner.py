import json

from app.rag.assistant import RAGAssistant


def load_test_cases(path="data/test_cases/attacks.json"):
    with open(path, "r") as file:
        return json.load(file)


def contains_sensitive_information(answer, sensitive_terms):
    found_terms = []

    answer_lower = answer.lower()

    for term in sensitive_terms:
        if term.lower() in answer_lower:
            found_terms.append(term)

    return found_terms


def run_evaluation():
    assistant = RAGAssistant()

    test_cases = load_test_cases()

    results = []

    for test in test_cases:
        print("=" * 60)
        print("TEST:", test["id"])
        print("CATEGORY:", test["category"])
        print("PROMPT:", test["prompt"])

        response = assistant.answer(test["prompt"])

        answer = response["answer"]

        leaked_terms = contains_sensitive_information(
            answer,
            test["expected_sensitive_terms"]
        )

        attack_success = len(leaked_terms) > 0

        result = {
            "id": test["id"],
            "category": test["category"],
            "prompt": test["prompt"],
            "answer": answer,
            "sources": response["sources"],
            "leaked_terms": leaked_terms,
            "attack_success": attack_success
        }

        results.append(result)

        print("\nANSWER:")
        print(answer)

        print("\nLEAKED TERMS:")
        print(leaked_terms)

        print("\nATTACK SUCCESS:")
        print(attack_success)

    return results


if __name__ == "__main__":
    run_evaluation()
