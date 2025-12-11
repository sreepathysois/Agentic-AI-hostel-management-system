import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nl_agent import generate_sql_and_run  # existing endpoint for /api/query
from chat_agent import orchestrate_message

app = FastAPI(title="Agentic HMS - BI + Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    message: str
    debug: Optional[bool] = False

@app.post("/api/query")
async def query(req: QueryRequest):
    if not req.question or len(req.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="question required")
    res = generate_sql_and_run(req.question.strip())
    return res

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message or len(req.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="message required")
    res = orchestrate_message(req.message.strip(), debug=bool(req.debug))
    return res

@app.get("/")
async def root():
    return {"status": "ok"}

