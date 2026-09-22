from app.rag.loader import load_documents
from app.rag.embeddings import EmbeddingModel

docs = load_documents()

texts = [doc["text"] for doc in docs if doc["text"].strip()]

model = EmbeddingModel()

embeddings = model.encode(texts)

print("Number of non-empty documents:", len(texts))
print("Number of embeddings:", len(embeddings))

if embeddings:
    print("Embedding dimension:", len(embeddings[0]))
