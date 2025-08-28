import os
from typing import List, Dict

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

_client = OpenAI()
_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma.get_or_create_collection("book_summaries")

def _embed(text: str) -> list[float]:
    text = text.replace("\n", " ")
    return _client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def search(query: str, k: int = 3) -> List[Dict]:
    """Search the vector store using the provided query string."""
    embedding = _embed(query)
    results = _collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["metadatas", "distances"]
    )

    output = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        output.append({
            "title": meta.get("title"),
            "distance": float(dist),
            "metadata": meta
        })

    return output
