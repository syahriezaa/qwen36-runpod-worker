import os
import random
import shutil
import signal
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Union

import requests
import runpod

LLAMA_BIN = os.getenv("LLAMA_BIN", "/opt/llama/bin/llama-server")
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/runpod-volume/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
)
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "qwen36-a3b")
LLAMA_HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8080"))

N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "999"))
CTX_SIZE = int(os.getenv("CTX_SIZE", "32768"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2048"))
UBATCH_SIZE = int(os.getenv("UBATCH_SIZE", "2048"))
PARALLEL_SLOTS = int(os.getenv("PARALLEL_SLOTS", "4"))
LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", "600"))
RETRY_MAX_TRIES = int(os.getenv("RETRY_MAX_TRIES", "8"))
RETRY_BASE_MS = int(os.getenv("RETRY_BASE_MS", "120"))
MAX_LOCAL_PARALLEL = int(os.getenv("MAX_LOCAL_PARALLEL", str(PARALLEL_SLOTS or 4)))

LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1"
LLAMA_LOG_PATH = Path(os.getenv("LLAMA_LOG_PATH", "/tmp/llama-server.log"))

_llama_proc: subprocess.Popen | None = None
_start_lock = threading.Lock()


def _wait_port(host: str, port: int, deadline_s: int = 120) -> None:
    started_at = time.time()
    while time.time() - started_at < deadline_s:
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return
        except OSError:
            if _llama_proc is not None and _llama_proc.poll() is not None:
                raise RuntimeError(_format_llama_failure(_llama_proc.returncode))
            time.sleep(0.2)
    raise RuntimeError(
        f"[llama] port {host}:{port} did not open within {deadline_s}s. {_format_llama_failure(None)}"
    )


def _tail_llama_log(max_lines: int = 80) -> str:
    if not LLAMA_LOG_PATH.exists():
        return "llama log file not found"

    try:
        lines = LLAMA_LOG_PATH.read_text(errors="replace").splitlines()
    except Exception as err:
        return f"unable to read llama log: {err}"

    if not lines:
        return "llama log is empty"

    return "\n".join(lines[-max_lines:])


def _format_llama_failure(returncode: int | None) -> str:
    rc_part = "still running" if returncode is None else f"exit code {returncode}"
    return f"llama-server {rc_part}. log tail:\n{_tail_llama_log()}"


def _models_ready(timeout_s: int = 300) -> str | None:
    started_at = time.time()
    while time.time() - started_at < timeout_s:
        try:
            response = requests.get(f"{LLAMA_BASE_URL}/models", timeout=2)
            if response.status_code == 503:
                time.sleep(0.4)
                continue
            if response.status_code == 200:
                data = response.json().get("data", [])
                if data:
                    match = next((item for item in data if item.get("id") == MODEL_ALIAS), data[0])
                    return match.get("id")
        except Exception:
            pass
        time.sleep(0.4)
    return None


def _post_with_retry(path: str, payload: Dict[str, Any], timeout_s: int) -> requests.Response:
    url = f"{LLAMA_BASE_URL}/{path}"
    last_error = None

    for attempt in range(RETRY_MAX_TRIES):
        response = None
        try:
            response = requests.post(url, json=payload, timeout=timeout_s)
            if response.status_code in (409, 429, 503):
                raise requests.HTTPError(f"transient status {response.status_code}", response=response)
            response.raise_for_status()
            return response
        except Exception as err:
            last_error = err
            if response is not None and 400 <= response.status_code < 500 and response.status_code not in (409, 429):
                raise
            delay = (RETRY_BASE_MS / 1000.0) * (2**attempt) * (0.75 + 0.5 * random.random())
            time.sleep(min(delay, 5.0))

    raise RuntimeError(f"request to {url} failed after retries: {last_error}")


def start_llama_server() -> None:
    global _llama_proc

    with _start_lock:
        if _llama_proc is not None and _llama_proc.poll() is None:
            return

        llama_bin = LLAMA_BIN
        if not os.path.isabs(llama_bin):
            resolved = shutil.which(llama_bin)
            if resolved:
                llama_bin = resolved

        wait_started = time.time()
        while not (os.path.isfile(llama_bin) and os.access(llama_bin, os.X_OK)):
            if time.time() - wait_started > 60:
                raise RuntimeError(f"[llama] LLAMA_BIN not found or not executable: {llama_bin}")
            if not os.path.isabs(LLAMA_BIN):
                resolved = shutil.which(LLAMA_BIN)
                if resolved:
                    llama_bin = resolved
            time.sleep(1)

        cmd = [
            llama_bin,
            "-m",
            MODEL_PATH,
            "--host",
            "0.0.0.0",
            "--port",
            str(LLAMA_PORT),
            "--jinja",
            "-ngl",
            str(N_GPU_LAYERS),
            "--ctx-size",
            str(CTX_SIZE),
            "-b",
            str(BATCH_SIZE),
            "-ub",
            str(UBATCH_SIZE),
            "-fa",
            "on",
            "--parallel",
            str(PARALLEL_SLOTS),
            "--no-webui",
            "--alias",
            MODEL_ALIAS,
        ]
        print(f"[llama] starting server: {' '.join(cmd)}", flush=True)
        LLAMA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LLAMA_LOG_PATH.write_text("")
        log_handle = open(LLAMA_LOG_PATH, "a", encoding="utf-8")
        _llama_proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True)

        _wait_port(LLAMA_HOST, LLAMA_PORT, deadline_s=120)
        model_id = _models_ready(timeout_s=300)
        if not model_id:
            raise RuntimeError(f"[llama] model did not become ready within 300s. {_format_llama_failure(None)}")

        try:
            _post_with_retry(
                "completions",
                {
                    "model": model_id,
                    "prompt": "ping",
                    "max_tokens": 1,
                    "top_k": 40,
                    "top_p": 0.9,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout_s=10,
            )
        except Exception:
            pass


def stop_llama_server() -> None:
    global _llama_proc
    if _llama_proc is None or _llama_proc.poll() is not None:
        return

    _llama_proc.send_signal(signal.SIGTERM)
    try:
        _llama_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _llama_proc.kill()
    _llama_proc = None


def _complete_one(
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    response = _post_with_retry(
        "completions",
        {
            "model": MODEL_ALIAS,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": int(top_k),
            "repeat_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "stream": False,
        },
        LLAMA_TIMEOUT,
    )
    return response.json()["choices"][0]["text"].strip()


def call_llama(
    messages: Union[str, List[str], List[Dict[str, str]]],
    temperature: float = 0.6,
    max_tokens: int = 512,
    top_p: float = 0.95,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
) -> Union[str, List[str]]:
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        response = _post_with_retry(
            "chat/completions",
            {
                "model": MODEL_ALIAS,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": max(1, int(top_k)),
                "repeat_penalty": repetition_penalty,
                "max_tokens": max_tokens,
                "stream": False,
            },
            LLAMA_TIMEOUT,
        )
        return response.json()["choices"][0]["message"]["content"]

    if isinstance(messages, str):
        return _complete_one(messages, temperature, max_tokens, top_p, top_k, repetition_penalty)

    prompts = list(messages)
    if not prompts:
        return []

    results: List[str] = ["" for _ in prompts]
    max_workers = min(len(prompts), MAX_LOCAL_PARALLEL or len(prompts))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _complete_one,
                prompts[index],
                temperature,
                max_tokens,
                top_p,
                top_k,
                repetition_penalty,
            ): index
            for index in range(len(prompts))
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    start_llama_server()
    payload = event["input"]
    answers = call_llama(
        messages=payload["query"],
        temperature=payload["temperature"],
        max_tokens=payload["max_tokens"],
        top_p=payload["top_p"],
        top_k=payload["top_k"],
        repetition_penalty=payload["repetition_penalty"],
    )
    return {"text": answers}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
