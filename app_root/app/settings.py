import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY is missing. Please check your .env file.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
