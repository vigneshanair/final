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
    assistant = RAGAssistant(protected=True)

    test_cases = load_test_cases()

    successful_attacks = 0

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

        if attack_success:
            successful_attacks += 1

        print("\nRAW ANSWER:")
        print(response["raw_answer"])

        print("\nPROTECTED ANSWER:")
        print(answer)

        print("\nDETECTED:")
        print(response["detected_sensitive_terms"])

        print("\nLEAKED TERMS AFTER PROTECTION:")
        print(leaked_terms)

        print("\nATTACK SUCCESS:")
        print(attack_success)

    total_tests = len(test_cases)

    attack_success_rate = (
        successful_attacks / total_tests * 100
        if total_tests > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print("Total attacks:", total_tests)
    print("Successful attacks:", successful_attacks)
    print(f"Attack Success Rate: {attack_success_rate:.2f}%")


if __name__ == "__main__":
    run_evaluation()
