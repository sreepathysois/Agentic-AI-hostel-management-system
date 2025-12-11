# make_schema_pretext.py
import json
import os
from pathlib import Path
from collections import OrderedDict

SRC = os.environ.get("SCHEMA_JSON_PATH", "schema_export.json")
OUT_PROMPT = os.environ.get("SCHEMA_PRETEXT", "schema_pretext.txt")
OUT_META = os.environ.get("SCHEMA_META", "schema_metadata.json")

def summarize_schema(schema: dict, max_cols_preview=8):
    """
    Produce a compact text summary suitable for system prompts.
    """
    lines = []
    lines.append("Database schema summary (compact). Use exact table and column names shown.")
    lines.append("Do NOT invent columns. Only use these tables and columns.")
    lines.append("----")
    table_index = OrderedDict()

    for tname, tinfo in schema.items():
        cols = tinfo.get("columns", [])
        fks = tinfo.get("foreign_keys", [])
        # pick first N columns as preview
        preview_cols = cols[:max_cols_preview]
        col_lines = []
        for c in preview_cols:
            cname = c.get("name")
            ctype = c.get("type")
            nullable = c.get("nullable")
            col_lines.append(f"{cname} ({ctype}, nullable={nullable})")
        lines.append(f"Table: {tname}")
        for cl in col_lines:
            lines.append("  - " + cl)
        if len(cols) > max_cols_preview:
            lines.append(f"  - ... (+{len(cols)-max_cols_preview} more columns)")
        if fks:
            lines.append("  Foreign keys:")
            for fk in fks:
                # fk format: {'constrained_columns': [...], 'referred_table': 'xyz', ...}
                constrained = fk.get("constrained_columns", [])
                ref_table = fk.get("referred_table")
                ref_cols = fk.get("referred_columns", [])
                lines.append(f"    {constrained} -> {ref_table}({ref_cols})")
        lines.append("")  # blank line
        # add to index
        table_index[tname] = {
            "columns": [c.get("name") for c in cols],
            "fk_count": len(fks)
        }
    return "\n".join(lines), table_index

def main():
    p = Path(SRC)
    if not p.exists():
        print(f"Schema file not found at {SRC}. Set SCHEMA_JSON_PATH env var or place schema_export.json here.")
        return

    schema = json.loads(p.read_text())
    summary_text, index = summarize_schema(schema)

    # add rules that BI agent must follow
    rules = [
        "",
        "=== RULES FOR BI AGENT ===",
        "1) ONLY produce parameterized SELECT SQL queries. No INSERT/UPDATE/DELETE/ALTER/DROP.",
        "2) Use only the tables and columns listed above.",
        "3) If the user asks to modify data, respond: 'This requires the Management Agent. I can only generate SELECT queries.'",
        "4) If ambiguous, ask for exactly one clarifying parameter (date range, hostel block, student id, etc.).",
        "5) Always return the SQL wrapped in ```sql ... ``` when responding with SQL.",
        ""
    ]
    final_text = summary_text + "\n".join(rules)

    Path(OUT_PROMPT).write_text(final_text)
    Path(OUT_META).write_text(json.dumps(index, indent=2))
    print(f"Wrote compact schema prompt to {OUT_PROMPT}")
    print(f"Wrote schema index to {OUT_META}")

if __name__ == "__main__":
    main()

