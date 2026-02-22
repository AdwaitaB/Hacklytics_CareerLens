import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from cortex import CortexClient, DistanceMetric
import uvicorn

app = FastAPI(title="CareerLens AI Backend")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config (keys from environment variables — never hardcoded) ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ACTIAN_HOST = os.environ.get("ACTIAN_HOST", "localhost:50051")
COLLECTION_NAME = "jobs"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set!")

# ── Request Models ──
class AskRequest(BaseModel):
    question: str
    profession: str = ""
    year: int = 2024
    metric: str = "salary"

class VectorSearchRequest(BaseModel):
    profession: str
    year: int = 2024
    top_k: int = 5

class EmbedRequest(BaseModel):
    text: str


# ══════════════════════════════════════════════════════════
# ROUTE: Health Check
# ══════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "CareerLens AI backend is running ✅"}


# ══════════════════════════════════════════════════════════
# ROUTE: Ask AI (Gemini)
# ══════════════════════════════════════════════════════════

@app.post("/ask")
async def ask_gemini(req: AskRequest):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        system_prompt = f"""You are a U.S. workforce intelligence analyst with deep expertise in 
Bureau of Labor Statistics (BLS) data, occupational employment statistics, and labor market trends 
from 2005 to 2030. You have access to data covering all 50 U.S. states across 10 professions.

Current user context:
- Selected profession: {req.profession or 'General workforce'}
- Selected year: {req.year}
- Primary metric of interest: {req.metric}

Your responses should be:
- Data-driven and specific (use real salary figures, percentages, state names)
- Analytical and insightful, not generic
- Structured with clear sections when answering complex questions
- Around 200-350 words — detailed but concise
- Written in a professional analyst tone

Focus on BLS occupational data, state-level salary divergence, automation risk trends, 
demand growth patterns, and career trajectory insights."""

        full_prompt = f"{system_prompt}\n\nUser question: {req.question}"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt
        )

        return {
            "answer": response.text,
            "profession": req.profession,
            "year": req.year,
            "source": "Gemini API (Live)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


# ══════════════════════════════════════════════════════════
# ROUTE: Vector Search (Actian VectorAI)
# ══════════════════════════════════════════════════════════

@app.post("/vector-search")
async def vector_search(req: VectorSearchRequest):
    try:
        query_text = f"Job profession: {req.profession}. Year: {req.year}. Career trajectory and salary growth patterns."

        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        embed_result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query_text]
        )
        query_vector = list(embed_result.embeddings[0].values)

        with CortexClient(ACTIAN_HOST) as cortex:
            results = cortex.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=req.top_k
            )

        formatted = []
        for i, r in enumerate(results):
            formatted.append({
                "rank": i + 1,
                "job_title": r.payload.get("job_title", "Unknown"),
                "description": r.payload.get("description", ""),
                "similarity_score": round(r.score, 4),
                "similarity_pct": round(r.score * 100, 1)
            })

        return {
            "query": query_text,
            "profession": req.profession,
            "year": req.year,
            "results": formatted,
            "source": "Actian VectorAI DB (Live)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")


# ══════════════════════════════════════════════════════════
# ROUTE: Embed a single text (utility)
# ══════════════════════════════════════════════════════════

@app.post("/embed")
async def embed_text(req: EmbedRequest):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[req.text]
        )
        vector = list(result.embeddings[0].values)
        return {"text": req.text, "dimension": len(vector), "embedding": vector[:5], "note": "Showing first 5 dims only"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

