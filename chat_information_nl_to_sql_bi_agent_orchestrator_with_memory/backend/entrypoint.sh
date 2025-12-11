#!/usr/bin/env bash
set -euo pipefail

# Optionally reindex knowledge base at startup
if [ "${REINDEX_KB:-true}" = "true" ]; then
  echo "Reindexing KB..."
  /app/scripts/reindex_kb.sh /app/knowledge_base || echo "reindex failed, continuing..."
fi

exec "$@"

