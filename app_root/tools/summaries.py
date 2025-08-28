import json
from pathlib import Path
from functools import lru_cache

# Location of the dataset
DATA_FILE = Path(__file__).resolve().parents[1] / "app" / "data" / "book_summaries.json"

@lru_cache(maxsize=1)
def _index():
    """Load the summaries from the dataset only once (cached)."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["title"].strip().lower(): item.get("full_summary", "") for item in items}

def get_summary_by_title(title: str) -> str:
    """Return the full summary for the given title (case-insensitive)."""
    return _index().get(title.strip().lower(), "")
