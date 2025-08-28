import json
import os
from pathlib import Path
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DATA_PATH = Path("data/book_summaries.json")

client = OpenAI()

def _embed(text: str) -> list[float]:
    text = text.replace("\n", " ")
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def build():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    with DATA_FILE.open("r", encoding="utf-8") as f:
        items = json.load(f)

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    col = chroma.get_or_create_collection("book_summaries")

# Clear and rebuild for idempotency
    try:
        col.delete(where={})
    except Exception:
        pass

    ids, docs, metas, embs = [], [], [], []
    for i, it in enumerate(items):
        title = it["title"]
        themes = ", ".join(it.get("themes", []))
        short = it.get("short_summary", "")
        doc_text = f"{title}\nThemes: {themes}\n{short}"
        ids.append(f"b{i}")
        docs.append(doc_text)
        metas.append({
            "title": title,
            "themes": it.get("themes", []),
            "short_summary": short,
            "full_summary": it.get("full_summary", "")
        })
        embs.append(_embed(doc_text))

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"✅ Built {len(ids)} items into collection 'book_summaries' at {CHROMA_PATH}")

if __name__ == "__main__":
    DATA_FILE = DATA_PATH
    build()