#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Geti Tune FastAPI server
#
# Features:
# - Optionally seed the database before starting the server by setting:
#     SEED_DB=true
#
# Usage:
#   SEED_DB=true ./run.sh       # Seed database before launching server (make sure to pre-upload model artifacts)
#   ./run.sh                    # Run server without seeding
#
# Environment variables:
#   SEED_DB       If set to "true", runs `uv run app/cli seed` before starting the server.
#   APP_MODULE    Python module to run (default: app/main.py)
#   UV_CMD        Command to launch Uvicorn (default: "uv run")
#
# Requirements:
# - 'uv' CLI tool (Uvicorn) installed and available in PATH
# - Python modules and dependencies installed correctly
# -----------------------------------------------------------------------------

SEED_DB=${SEED_DB:-false}
APP_MODULE=${APP_MODULE:-src/main.py}
UV_CMD=${UV_CMD:-uv run}

export PYTHONUNBUFFERED=1
export PYTHONPATH=.:src

if [[ "$SEED_DB" == "true" ]]; then
  echo "Seeding the database..."
  $UV_CMD src/cli.py init-db
  $UV_CMD src/cli.py seed --with-model=True
fi

echo "Starting FastAPI server..."

exec $UV_CMD "$APP_MODULE"
