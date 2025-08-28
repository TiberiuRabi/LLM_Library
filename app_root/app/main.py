from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from app import settings
from rag import retriever
from tools.summaries import get_summary_by_title
from fastapi import HTTPException
import traceback

app = FastAPI(title="Smart Librarian (FastAPI)")
_client = OpenAI(api_key=settings.OPENAI_API_KEY)

class RecommendRequest(BaseModel):
    query: str
    k: int = 3

class RecommendResponse(BaseModel):
    recommended_title: str
    message: str
    lternatives: List[str]


@app.get("/health")
def health():
    return {"ok": True}

def _ask_llm_to_choose(query: str, candidates: List[dict]) -> dict:
# Ask the LLM to choose the best title and explain briefly (Romanian-friendly)
    titles = [c["title"] for c in candidates]
    sys = (
    "Ești un bibliotecar prietenos. Ai primit o întrebare și o listă de cărți candidate. "
    "Alege o singură carte care se potrivește cel mai bine cu tema, apoi răspunde cu JSON: "
    "{\"title\": str, \"why\": str}. Răspuns scurt (2-4 propoziții)."
    )
    usr = {
    "role": "user",
    "content": f"Întrebare: {query}\nCandidaturi: {titles}"
    }
    chat = _client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": sys}, usr],
        temperature=0.5,
    )
    import json

    try:
        data = json.loads(chat.choices[0].message.content)
        if not data.get("title"):
            raise ValueError("Missing title")
        return data
    except Exception as e:
    # Fallback: pick the first candidate
        print(f"Error occurred: {e}")
        return {"title": titles[0], "why": "Se potrivește cel mai bine dintre rezultatele recuperate."}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        # 1. Semantic search
        hits = retriever.search(req.query, k=req.k)
        if not hits:
            raise HTTPException(
                status_code=404,
                detail="Nicio potrivire găsită. Încearcă o altă temă sau adaugă mai multe cărți în dataset."
            )

        # 2. Ask LLM to choose a match
        choice = _ask_llm_to_choose(req.query, hits)
        title = choice.get("title")
        why = choice.get("why", "")

        if not title:
            raise HTTPException(
                status_code=500,
                detail="Răspunsul de la LLM nu a conținut un titlu valid."
            )

        # 3. Get full summary
        full = get_summary_by_title(title)
        if not full:
            # Try fallback from metadata
            for h in hits:
                if h["title"].lower() == title.lower():
                    full = h["metadata"].get("full_summary", "")
                    break

        # 4. Format the message
        message = (
            f"Îți recomand **{title}**. {why}\n\n"
            + (f"**Rezumat complet:** {full}" if full else "(Nu am găsit rezumatul complet pentru această carte.)")
        )

        alts = [h["title"] for h in hits if h["title"].lower() != title.lower()]

        return RecommendResponse(
            recommended_title=title,
            message=message,
            alternatives=alts,
        )

    except HTTPException:
        raise  # re-raise known FastAPI errors without wrapping

    except Exception as e:
        traceback.print_exc()  # log full stack trace in terminal
        raise HTTPException(
            status_code=500,
            detail=f"Eroare internă: {str(e)}"
        )