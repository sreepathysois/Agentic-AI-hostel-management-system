# backend/app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nl_agent import generate_sql_and_run

app = FastAPI(title="BI NL->SQL Agent API")

# Allow CORS from frontend container
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/api/query")
async def query(req: QueryRequest):
    if not req.question or len(req.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="question required")
    res = generate_sql_and_run(req.question.strip())
    return res

@app.get("/")
async def root():
    return {"status": "ok"}

