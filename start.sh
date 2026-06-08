#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting Qwen RunPod handler"
exec python3 -u /workspace/rp_handler.py
