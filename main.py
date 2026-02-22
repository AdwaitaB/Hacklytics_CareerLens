import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import uvicorn

app = FastAPI(title="CareerLens AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set!")

class AskRequest(BaseModel):
    question: str
    profession: str = ""
    year: int = 2024
    metric: str = "salary"

class VectorSearchRequest(BaseModel):
    profession: str
    year: int = 2024
    top_k: int = 5

@app.get("/")
def root():
    return {"status": "CareerLens AI backend is running ✅"}

@app.post("/ask")
async def ask_gemini(req: AskRequest):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""You are a U.S. workforce intelligence analyst. Profession: {req.profession}, Year: {req.year}, Metric: {req.metric}. Answer in 200-300 words with specific data.\n\nQuestion: {req.question}"""
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return {"answer": response.text, "profession": req.profession, "year": req.year}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-search")
async def vector_search(req: VectorSearchRequest):
    # Demo results since Actian is not connected
    demo = [
        {"rank":1,"job_title":f"{req.profession} - High Growth State","description":"Strong salary trajectory with above-average demand growth","similarity_pct":94},
        {"rank":2,"job_title":f"{req.profession} - Emerging Market","description":"Below median salary but rapid growth indicators","similarity_pct":87},
        {"rank":3,"job_title":f"{req.profession} - Stable Region","description":"Consistent employment, low volatility","similarity_pct":79},
        {"rank":4,"job_title":f"{req.profession} - Remote Hub","description":"High remote feasibility, coastal premium","similarity_pct":71},
        {"rank":5,"job_title":f"{req.profession} - Declining Market","description":"Automation pressure, retraining recommended","similarity_pct":63},
    ]
    return {"results": demo, "profession": req.profession, "year": req.year}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
