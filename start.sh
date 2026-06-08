#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${LLAMA_BIN:-}" || "${LLAMA_BIN}" == "llama-server" ]]; then
  for candidate in \
    "$(command -v llama-server 2>/dev/null || true)" \
    /usr/local/bin/llama-server \
    /llama.cpp/build/bin/llama-server \
    /app/llama-server \
    /opt/llama/bin/llama-server
  do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      export LLAMA_BIN="${candidate}"
      break
    fi
  done
fi

echo "[entrypoint] using llama binary: ${LLAMA_BIN:-<unset>}"
echo "[entrypoint] starting Qwen RunPod handler"
exec python3 -u /workspace/rp_handler.py
