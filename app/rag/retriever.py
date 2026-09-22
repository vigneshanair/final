import chromadb

from app.rag.loader import load_documents
from app.rag.embeddings import EmbeddingModel


class Retriever:
    def __init__(self):
        self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="deepshield_documents"
        )

        self.embedding_model = EmbeddingModel()

        self._load_into_database()

    def _load_into_database(self):
        docs = load_documents()

        docs = [doc for doc in docs if doc["text"].strip()]

        if not docs:
            return

        ids = [doc["id"] for doc in docs]
        texts = [doc["text"] for doc in docs]

        embeddings = self.embedding_model.encode(texts)

        metadatas = [
            {"source": doc["source"]}
            for doc in docs
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query, n_results=2):
        query_embedding = self.embedding_model.encode([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results
