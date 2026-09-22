from app.rag.assistant import RAGAssistant


assistant = RAGAssistant()

question = "What is Project Falcon?"

result = assistant.answer(question)

print("\nQUESTION:")
print(result["question"])

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print("-", source)
