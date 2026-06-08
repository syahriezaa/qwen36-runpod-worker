FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      tini \
      python3 \
      python3-pip \
      libgomp1 \
      libstdc++6 \
      libatomic1 \
      libcurl4t64 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
COPY rp_handler.py /workspace/rp_handler.py
COPY start.sh /usr/local/bin/start.sh

RUN sed -i 's/\r$//' /usr/local/bin/start.sh \
  && chmod +x /usr/local/bin/start.sh \
  && mkdir -p /opt/llama/bin \
  && LLAMA_SERVER_PATH="$(command -v llama-server)" \
  && ln -sf "${LLAMA_SERVER_PATH}" /opt/llama/bin/llama-server

ENV LLAMA_BIN=/opt/llama/bin/llama-server \
    MODEL_PATH=/runpod-volume/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    MODEL_ALIAS=qwen36-a3b \
    N_GPU_LAYERS=999 \
    CTX_SIZE=32768 \
    BATCH_SIZE=2048 \
    UBATCH_SIZE=2048 \
    PARALLEL_SLOTS=4 \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8080

ENTRYPOINT ["tini", "-s", "--"]
CMD ["/usr/local/bin/start.sh"]
