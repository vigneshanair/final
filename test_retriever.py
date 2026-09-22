from app.rag.retriever import Retriever

retriever = Retriever()

query = "What is Project Falcon?"

results = retriever.search(query)

print("\nQUESTION:")
print(query)

print("\nRESULTS:")

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i, document in enumerate(documents):
    print("\nResult", i + 1)
    print("Source:", metadatas[i]["source"])
    print("Distance:", distances[i])
    print("Text:")
    print(document)
    print("-" * 50)
