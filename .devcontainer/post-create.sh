#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

echo "==> Versions"
bun --version || true
go version || true
python3 --version || true
uv --version || true
gh --version | head -n1 || true

echo "==> ts/ : bun install"
if [ -d ts ]; then
  (cd ts && bun install)
fi

echo "==> python/langchain/ : uv sync"
if [ -d python/langchain ]; then
  (cd python/langchain && uv sync)
fi

echo "==> go/ : go mod download"
if [ -d go ]; then
  (cd go && go mod download)
fi

echo "Done."
