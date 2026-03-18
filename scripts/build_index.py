from pathlib import Path
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATASET_NAME = "lavita/MedQuAD"
SAVE_DIR = Path("./storage/faiss_medquad")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def row_to_document(row: dict) -> Document:
    question = row.get("question", "").strip()
    answer = row.get("answer", "").strip()

    source = (
        row.get("source")
        or row.get("source_name")
        or row.get("document_source")
        or row.get("origin")
        or "unknown"
    )

    focus = (
        row.get("focus")
        or row.get("topic")
        or row.get("disease")
        or row.get("subject")
        or "unknown"
    )

    qtype = row.get("qtype") or row.get("question_type") or "unknown"

    content = f"Question: {question}\nAnswer: {answer}"

    metadata = {
        "question": question,
        "source": str(source),
        "focus": str(focus),
        "qtype": str(qtype),
    }

    return Document(page_content=content, metadata=metadata)


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")

    print("Dataset loaded.")
    print(f"Total rows: {len(dataset)}")

    print("Sample columns:")
    print(dataset.column_names)

    if len(dataset) > 0:
        print("Sample row:")
        print(dataset[0])

    docs = []
    skipped = 0

    for row in dataset:
        question = row.get("question", "")
        answer = row.get("answer", "")

        if not question or not answer:
            skipped += 1
            continue

        docs.append(row_to_document(row))

    print(f"Valid documents: {len(docs)}")
    print(f"Skipped rows: {skipped}")

    print("Building embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(SAVE_DIR))

    print(f"Index saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()