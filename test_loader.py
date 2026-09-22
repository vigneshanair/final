from app.rag.loader import load_documents

docs = load_documents()

print(f"Loaded {len(docs)} documents")

for doc in docs:
    print("SOURCE:", doc["source"])
    print("TEXT:", doc["text"][:100])
    print("-" * 40)
