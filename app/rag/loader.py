from pathlib import Path


def load_documents(folder="data/documents"):
    documents = []

    for file in Path(folder).glob("*.txt"):
        text = file.read_text()

        documents.append({
            "id": file.stem,
            "text": text,
            "source": file.name
        })

    return documents
