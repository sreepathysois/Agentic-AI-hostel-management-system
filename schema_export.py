# schema_export.py
import json
import sqlalchemy
from sqlalchemy import create_engine, inspect
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

load_dotenv()

# Example MSSQL connection via pyodbc DSNless:
user = os.environ['DB_USER']
pwd  = os.environ['DB_PASSWORD']
host = os.environ['DB_HOST']
db   = os.environ['DB_NAME']

odbc_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={host};DATABASE={db};UID={user};PWD={pwd};"
    "Encrypt=yes;TrustServerCertificate=yes;"
)

conn_str = "mssql+pyodbc:///?odbc_connect=%s" % quote_plus(odbc_str)
engine = create_engine(conn_str, fast_executemany=True)

#conn_str = "mssql+pyodbc://%s:%s@%s/%s?driver=ODBC+Driver+18+for+SQL+Server" % (
#    quote_plus(user), quote_plus(pwd), host, db
#)

engine = create_engine(conn_str, fast_executemany=True)
insp = inspect(engine)

schema = {}
for table_name in insp.get_table_names():
    cols = []
    for col in insp.get_columns(table_name):
        cols.append({
            "name": col['name'],
            "type": str(col['type']),
            "nullable": col['nullable'],
            "default": col.get('default')
        })
    fks = []
    for fk in insp.get_foreign_keys(table_name):
        fks.append(fk)
    schema[table_name] = {"columns": cols, "foreign_keys": fks}

with open("schema_export.json", "w") as f:
    json.dump(schema, f, indent=2)

print("Wrote schema_export.json")

