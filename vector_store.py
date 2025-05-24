from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import json
from tqdm import tqdm

def create_vectorstore():
    # Load processed docs
    docs = []
    with open("data/processed/kcc_docs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Show progress while computing embeddings
    print("Generating embeddings and adding to vectorstore...")
    persist_dir = "data/chroma_db"

    vectordb = Chroma.from_texts(
        texts=list(tqdm(texts, desc="Embedding texts")),
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=persist_dir
    )

    vectordb.persist()
    print("✅ Vectorstore created and persisted!")

if __name__ == "__main__":
    create_vectorstore()
