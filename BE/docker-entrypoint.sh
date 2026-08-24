#!/bin/bash
set -e

echo "=== Graph RAG Backend Container Initializing ==="

# Wait for PostgreSQL
if [ -n "$PG_HOST" ]; then
    echo "Waiting for PostgreSQL ($PG_HOST:${PG_PORT:-5432})..."
    until nc -z -v -w5 "$PG_HOST" "${PG_PORT:-5432}" 2>/dev/null; do
        echo "PostgreSQL is unavailable - sleeping 2s"
        sleep 2
    done
    echo "PostgreSQL is up!"
fi

# Run Alembic migrations
echo "Executing database migrations..."
alembic upgrade head

echo "Database migrations completed successfully."

# Start FastAPI Uvicorn server
echo "Starting FastAPI server with reload..."
exec uvicorn app.main:app \
    --host "${SERVER_HOST:-0.0.0.0}" \
    --port "${SERVER_PORT:-8000}" \
    --reload \
    --reload-dir /app/BE/app \
    --reload-dir /app/RAG_package/packages
