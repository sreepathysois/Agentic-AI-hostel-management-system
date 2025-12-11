#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
KB_FOLDER=${1:-knowledge_base}

echo "Re-indexing KB from $KB_FOLDER into Qdrant..."

$PYTHON - <<PY
import sys, os
sys.path.insert(0, "/app")   # IMPORTANT FIX

from kb_rag import ingest_kb_folder

ingest_kb_folder(kb_folder="$KB_FOLDER")
print("Reindex complete")
PY

