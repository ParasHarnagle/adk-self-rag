import os
import re
import sys
import uuid
import pickle
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # ✅ correct modern import

# --------------------------------------------------------------------
# ⚙️ Load environment and configuration
# --------------------------------------------------------------------
load_dotenv()

LITELLM_PROXY_API_KEY = os.getenv("LITELLM_PROXY_API_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index.index")
FAISS_META_PATH = os.path.splitext(FAISS_INDEX_PATH)[0] + "_meta.pkl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
NUM_THREADS = 8

os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

# --------------------------------------------------------------------
# 🔧 Helper Functions
# --------------------------------------------------------------------
def sanitize_namespace(file_path: str) -> str:
    """Creates a clean FAISS namespace based on filename."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ascii_namespace = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name)
    return ascii_namespace


def load_pdf_chunks(file_path: str) -> List[str]:
    """Loads and splits a PDF into overlapping text chunks."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    return chunks


def embed_documents(chunks: list):
    """Embeds text chunks using OpenRouter text-embedding-3-large."""

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-large",
        api_key=os.getenv("LITELLM_PROXY_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    texts = [doc.page_content for doc in chunks]
    metadata = [doc.metadata | {"text": doc.page_content} for doc in chunks]

    print("🧠 Embedding chunks in parallel with OpenRouter (text-embedding-3-large)...")

    # Compute first embedding to detect vector dimension dynamically
    first_vec = np.array(embeddings.embed_query(texts[0]), dtype="float32")
    dim = len(first_vec)
    vectors = np.zeros((len(texts), dim), dtype="float32")
    vectors[0] = first_vec

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def embed_one(i: int):
        if i == 0:
            return i, first_vec
        embedding = embeddings.embed_query(texts[i])
        return i, np.array(embedding, dtype="float32")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(embed_one, i) for i in range(1, len(texts))]
        for future in as_completed(futures):
            i, emb = future.result()
            vectors[i] = emb
    # print("-----------v---------",vectors)
    # print("-------m---------",metadata)
    return vectors, metadata


def save_faiss_index(vectors: np.ndarray, metadata: list, index_path: str, meta_path: str):
    """Saves embeddings + metadata into FAISS index and pickle file."""
    dim = vectors.shape[1]
    print(f"📦 Creating FAISS index with dimension: {dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved FAISS index to: {index_path}")
    print(f"✅ Saved metadata to: {meta_path}")
    print(f"📊 Total vectors: {len(metadata)}")


def append_to_existing_index(vectors: np.ndarray, metadata: list, index_path: str, meta_path: str):
    """Appends new embeddings to existing FAISS index and metadata."""
    print("📥 Loading existing FAISS index...")
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        existing_metadata = pickle.load(f)

    print("📈 Adding new vectors...")
    index.add(vectors)
    existing_metadata.extend(metadata)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(existing_metadata, f)

    print(f"✅ Appended {len(metadata)} new vectors. Total = {len(existing_metadata)}")


# --------------------------------------------------------------------
# 🚀 Main logic
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf_uploader.py /path/to/your/file.pdf")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    namespace = sanitize_namespace(file_path)
    print(f"📄 Loading PDF: {file_path}")
    chunks = load_pdf_chunks(file_path)
    print(f"🔗 Extracted {len(chunks)} text chunks.")

    if len(chunks) == 0:
        print("❌ No text chunks extracted.")
        return

    vectors, metadata = embed_documents(chunks)

    if os.path.exists(FAISS_INDEX_PATH):
        append_to_existing_index(vectors, metadata, FAISS_INDEX_PATH, FAISS_META_PATH)
    else:
        save_faiss_index(vectors, metadata, FAISS_INDEX_PATH, FAISS_META_PATH)

    print(f"🎉 Indexing complete for '{namespace}'")


if __name__ == "__main__":
    main()
