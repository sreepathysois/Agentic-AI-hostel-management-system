#!/usr/bin/env bash
set -e

echo "Waiting for Qdrant to be ready..."
sleep 5

echo "Reindexing Knowledge Base..."
./scripts/reindex_kb.sh || echo "Reindex failed — continuing startup."

echo "Starting backend..."
exec uvicorn app:app --host 0.0.0.0 --port 8000

