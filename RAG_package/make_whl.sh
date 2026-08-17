#!/usr/bin/env bash
set -e
ROOT="$(pwd)"
OUTDIR="$ROOT/dist-all"
mkdir -p "$OUTDIR"

for d in packages/*; do
  if [ -f "$d/pyproject.toml" ]; then
    echo "Building $d..."
    (cd "$d" && python -m build --wheel --outdir "$OUTDIR")
  else
    echo "Skip $d (no pyproject.toml)"
  fi
done

ls -la "$OUTDIR"
