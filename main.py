import os
import json
import sqlite3
import httpx
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# === Load .env ===
load_dotenv()
API_KEY = os.getenv("API_KEY")
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai"
DB_PATH = "knowledge_base.db"

# === FastAPI setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Pydantic models ===
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkItem(BaseModel):
    url: str
    text: str

class QuestionResponse(BaseModel):
    answer: str
    links: List[LinkItem]

# === Embed question ===
def embed_text(texts: List[str]) -> List[List[float]]:
    url = f"{AIPROXY_BASE_URL}/v1/embeddings"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

# === Cosine similarity ===
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-8)

# === DB retrieval ===
def find_similar_content(query_embedding: List[float], top_k: int = 5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content, source, embedding FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    scored = []
    for content, source, embedding_json in rows:
        db_embedding = json.loads(embedding_json)
        score = cosine_similarity(query_embedding, db_embedding)
        scored.append((score, content, source))

    scored.sort(reverse=True)
    return scored[:top_k]

# === GPT Answer ===
def answer_question(question: str, k: int = 5):
    question_vector = embed_text([question])[0]
    top_docs = find_similar_content(question_vector, top_k=k)
    context = "\n\n".join([doc[1] for doc in top_docs])

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful Virtual TA for the Tools for Data Science (TDS) course.\n"
                    "Use the context below to answer clearly:\n\n"
                    f"{context}\n\n"
                    "If context has source links, respond in JSON format like:\n"
                    "{ \"answer\": \"...\", \"links\": [ {\"url\": \"...\", \"text\": \"...\"}, ... ] }"
                )
            },
            {"role": "user", "content": question}
        ]
    }

    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{AIPROXY_BASE_URL}/v1/chat/completions"
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        raw_answer = result["choices"][0]["message"]["content"].strip()

    try:
        parsed = json.loads(raw_answer)
        answer = parsed["answer"]
        links = parsed.get("links", [])
    except Exception:
        answer = raw_answer
        links = [
            {"url": doc[2], "text": doc[1][:300]}
            for doc in top_docs
        ]

    return answer, links

# === API route ===
@app.post("/", response_model=QuestionResponse)
async def ask_question(data: QuestionRequest):
    try:
        answer, sources = answer_question(data.question)
        return JSONResponse(content={"answer": answer, "links": sources})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True) 