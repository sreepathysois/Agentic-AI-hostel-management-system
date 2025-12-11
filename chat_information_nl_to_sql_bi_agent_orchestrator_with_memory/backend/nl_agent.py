# nl_agent.py
import os
import re
import json
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text, inspect
import pandas as pd

# LLM wrapper import (your wrapper)
from langchain_openai import ChatOpenAI
from llm_utils import normalize_llm_response
from langchain_core.prompts import PromptTemplate

load_dotenv()

# DB env
user = os.environ.get("DB_USER")
pwd = os.environ.get("DB_PASSWORD")
host = os.environ.get("DB_HOST")
db   = os.environ.get("DB_NAME")
schema_pretext_path = os.environ.get("SCHEMA_PRETEXT", "schema_pretext.txt")

if not (user and pwd and host and db):
    raise SystemExit("Missing DB credentials in .env")

odbc_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={host};DATABASE={db};UID={user};PWD={pwd};"
    "Encrypt=yes;TrustServerCertificate=yes;"
)

conn_str = "mssql+pyodbc:///?odbc_connect=%s" % quote_plus(odbc_str)
engine = create_engine(conn_str, fast_executemany=True)

# load schema pretext (compact schema summary)
here = os.path.dirname(__file__)
sp = schema_pretext_path
if not os.path.isabs(sp):
    sp = (
        os.path.join(here, "..", sp)
        if os.path.exists(os.path.join(here, "..", sp))
        else os.path.join(here, sp)
    )
try:
    with open(sp, "r", encoding="utf-8") as f:
        SCHEMA_TEXT = f.read()
except Exception:
    SCHEMA_TEXT = "Schema summary not available."

# Prompt template (UPDATED to use memory_context)
PROMPT_TEMPLATE = f"""
You are a BI assistant generating SELECT-only SQL for MSSQL.

Here is a short JSON with known session context (may be empty):
{{memory_context}}

Use this context to fill in reasonable defaults when the user omits details.
For example, if session_block is known and the user asks "show available seats"
without specifying the block, default to that block in the WHERE clause.

Database schema (MSSQL):
{SCHEMA_TEXT}

User question: {{question}}

Rules:
1. Only return SQL wrapped in ```sql ... ```
2. No explanation, only SQL.
3. SELECT only.
4. If something is ambiguous and cannot be safely inferred even from the session context:
   - Ask ONE clear clarifying question instead of guessing.
5. If the user is asking to INSERT/UPDATE/DELETE or perform changes, respond
   exactly with: "This requires the Management Agent."
"""

PROMPT = PromptTemplate(
    input_variables=["question", "memory_context"],
    template=PROMPT_TEMPLATE,
)

# LLM init
llm = ChatOpenAI(
    model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    temperature=float(os.environ.get("LLM_TEMP", "0")),
)

def call_llm(prompt_text: str) -> str:
    """
    Call LLM and normalize to string.
    """
    try:
        try:
            resp = llm.invoke(prompt_text)
        except TypeError:
            resp = llm(prompt_text)
    except Exception as e:
        return f"LLM error: {e}"
    return normalize_llm_response(resp)

# Safety & helpers
def is_safe(sql: str) -> bool:
    bad = ["insert", "update", "delete", "drop", "alter", "truncate", "create", ";--"]
    s = (sql or "").lower()
    return not any(b in s for b in bad)

def extract_sql_from_text(resp_text: str) -> str:
    if not resp_text:
        return None
    m = re.search(r"```sql(.*?)```", resp_text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", resp_text, re.S)
    if m:
        return m.group(1).strip()
    # fallback: assume full text is SQL
    return resp_text.strip()

def generate_sql_and_run(question: str, memory_context: str = "") -> dict:
    """
    Main exported function: generate SQL (via LLM), check safety, run read-only SELECT.
    'memory_context' is a JSON string describing session-level facts
    (block, allergies, rollno, etc.) which the LLM can use as defaults.
    """
    prompt_text = PROMPT.format(
        question=question,
        memory_context=memory_context or "{}",
    )
    llm_raw = call_llm(prompt_text)
    sql = extract_sql_from_text(llm_raw)
    out = {
        "prompt": prompt_text,
        "llm_raw": llm_raw,
        "sql": sql,
        "safety_ok": None,
        "error": None,
        "data": None,
    }

    if not sql:
        out["error"] = "No SQL found in LLM response"
        return out

    out["safety_ok"] = is_safe(sql)
    if not out["safety_ok"]:
        out["error"] = "Unsafe SQL (write/DDL detected)"
        return out

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
            out["data"] = df.to_dict(orient="records")
    except Exception as e:
        out["error"] = str(e)

    return out

# allow importing SCHEMA_TEXT from other modules

