# backend/nl_agent.py
import os
import re
import json
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
import pandas as pd

# langchain imports depending on your installed lib
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

# DB env keys are read from .env at project root or backend/.env if you prefer
user = os.environ.get("DB_USER")
pwd = os.environ.get("DB_PASSWORD")
host = os.environ.get("DB_HOST")
db   = os.environ.get("DB_NAME")
schema_pretext_path = os.environ.get("SCHEMA_PRETEXT", "../schema_pretext.txt")

if not (user and pwd and host and db):
    raise SystemExit("Missing DB_USER / DB_PASSWORD / DB_HOST / DB_NAME in env")

# Build ODBC connection string (your exact format)
odbc_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={host};DATABASE={db};UID={user};PWD={pwd};"
    "Encrypt=yes;TrustServerCertificate=yes;"
)
conn_str = "mssql+pyodbc:///?odbc_connect=%s" % quote_plus(odbc_str)
engine = create_engine(conn_str, fast_executemany=True)

# Load compact schema pretext
with open(os.path.join(os.path.dirname(__file__), schema_pretext_path), "r", encoding="utf-8") as f:
    SCHEMA_TEXT = f.read()

PROMPT_TEMPLATE = f"""
You are a BI assistant that generates only parameterized SELECT SQL statements for a Microsoft SQL Server reporting database.
Follow the rules strictly.

{SCHEMA_TEXT}

User question: {{question}}

Rules:
1) ONLY return SQL wrapped in ```sql ... ```
2) Return ONLY the SQL block. No extra explanation.
3) SELECT only. No INSERT/UPDATE/DELETE/ALTER/DROP/TRUNCATE.
4) If ambiguous, ask one clarifying question.
5) If the user asks to modify data, reply exactly: "This requires the Management Agent."
"""
PROMPT = PromptTemplate(input_variables=["question"], template=PROMPT_TEMPLATE)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def call_llm(prompt_text: str) -> str:
    """Call the LLM and normalize return text (works across versions)."""
    try:
        resp = llm.invoke(prompt_text)
    except Exception:
        resp = llm(prompt_text)

    # try common fields
    if hasattr(resp, "content"):
        return resp.content
    if hasattr(resp, "generations"):
        gens = resp.generations
        try:
            return gens[0][0].text
        except Exception:
            pass
    if hasattr(resp, "text"):
        return resp.text
    return str(resp)

def extract_sql(resp: str) -> str | None:
    m = re.search(r"```sql(.*?)```", resp, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", resp, re.S)
    if m:
        return m.group(1).strip()
    return None

def is_safe(sql: str) -> bool:
    bad = ["insert","update","delete","drop","alter","truncate","create",";--"]
    return not any(b in sql.lower() for b in bad)

def generate_sql_and_run(question: str) -> dict:
    """
    Returns detailed steps and final results:
    {
      "prompt": "...",
      "llm_raw": "...",
      "sql": "...",
      "safety_ok": true/false,
      "error": "...",
      "data": [...]
    }
    """
    prompt = PROMPT.format(question=question)
    llm_raw = call_llm(prompt)
    sql = extract_sql(llm_raw)
    result = {
        "prompt": prompt,
        "llm_raw": llm_raw,
        "sql": sql,
        "safety_ok": None,
        "error": None,
        "data": None
    }
    if not sql:
        result["error"] = "No SQL returned by LLM."
        result["safety_ok"] = False
        return result

    ok = is_safe(sql)
    result["safety_ok"] = ok
    if not ok:
        result["error"] = "Generated SQL failed safety checks (non-SELECT or forbidden token)."
        return result

    # execute
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        result["data"] = df.to_dict(orient="records")
    except Exception as e:
        result["error"] = str(e)
    return result

