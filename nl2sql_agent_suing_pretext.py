# nl2sql_agent_using_pretext.py
import os
import re
import json
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
import pandas as pd

# Updated imports for new LangChain versions
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# -----------------------------------------------------------
# DB CREDENTIALS
# -----------------------------------------------------------
user = os.environ.get("DB_USER")
pwd = os.environ.get("DB_PASSWORD")
host = os.environ.get("DB_HOST")
db   = os.environ.get("DB_NAME")
schema_pretext_path = os.environ.get("SCHEMA_PRETEXT", "schema_pretext.txt")

if not (user and pwd and host and db):
    raise SystemExit("Missing DB credentials in .env")

# -----------------------------------------------------------
# ODBC CONNECTION (YOUR EXACT FORMAT)
# -----------------------------------------------------------
odbc_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={host};DATABASE={db};UID={user};PWD={pwd};"
    "Encrypt=yes;TrustServerCertificate=yes;"
)

conn_str = "mssql+pyodbc:///?odbc_connect=%s" % quote_plus(odbc_str)
engine = create_engine(conn_str, fast_executemany=True)

# -----------------------------------------------------------
# LOAD SCHEMA PRETEXT
# -----------------------------------------------------------
with open(schema_pretext_path, "r", encoding="utf-8") as f:
    SCHEMA_TEXT = f.read()

# -----------------------------------------------------------
# PROMPT TEMPLATE
# -----------------------------------------------------------
PROMPT_TEMPLATE = f"""
You are a BI assistant generating SELECT-only SQL for Microsoft SQL Server.

{SCHEMA_TEXT}

User question: {{question}}

Rules:
1. Only return SQL wrapped in ```sql ... ```
2. Return ONLY SQL. No explanation.
3. SELECT only. No INSERT/UPDATE/DELETE/ALTER/DROP/TRUNCATE.
4. If the question is ambiguous: ask ONE clarifying question.
5. If user asks to modify data: respond exactly "This requires the Management Agent."
"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=PROMPT_TEMPLATE
)

# -----------------------------------------------------------
# LLM SETUP (new LC API)
# -----------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -----------------------------------------------------------
# LLM CALL HANDLER (works across LC versions)
# -----------------------------------------------------------
def call_llm(prompt_text: str) -> str:
    """
    Handles all possible LangChain return types.
    """
    try:
        # Most LC versions allow calling the model directly
        resp = llm.invoke(prompt_text)
    except AttributeError:
        # fallback for older builds
        resp = llm(prompt_text)

    # resp is a BaseMessage-like -> .content has the text
    if hasattr(resp, "content"):
        return resp.content

    return str(resp)

# -----------------------------------------------------------
# SAFETY CHECKS
# -----------------------------------------------------------
def is_safe(sql: str) -> bool:
    bad = ["insert","update","delete","drop","alter","truncate","create",";--"]
    return not any(b in sql.lower() for b in bad)

def extract_sql(resp: str):
    """
    Pull SQL from ```sql ... ``` blocks.
    """
    m = re.search(r"```sql(.*?)```", resp, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", resp, re.S)
    if m:
        return m.group(1).strip()
    return None

# -----------------------------------------------------------
# MAIN EXECUTION FUNCTION
# -----------------------------------------------------------
def run_question(question: str):
    prompt = PROMPT.format(question=question)
    resp = call_llm(prompt)

    sql = extract_sql(resp)
    if not sql:
        return {"error": "No SQL found in LLM response", "raw": resp}

    if not is_safe(sql):
        return {"error": "Unsafe SQL detected", "sql": sql}

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return {"sql": sql, "data": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e), "sql": sql}

# -----------------------------------------------------------
# CLI LOOP
# -----------------------------------------------------------
if __name__ == "__main__":
    print("BI Agent Ready. Type a question or 'quit'.")
    while True:
        q = input("Question> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        print(json.dumps(run_question(q), indent=2, ensure_ascii=False))

