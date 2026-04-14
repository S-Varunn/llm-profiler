"""
Bridge vLLM's built-in torch profiler traces into this project's JSON output format.

Flow:
1) POST /start_profile on a vLLM server launched with --profiler-config
2) Send one or more /v1/chat/completions requests
3) POST /stop_profile
4) Load latest trace file from torch_profiler_dir
5) Analyze trace with TraceAnalyzer and save profiler-style JSON

Example:
  export VLLM_BASE_URL=http://127.0.0.1:8000
  export VLLM_MODEL=meta-llama/Llama-3-8B
  export VLLM_TRACE_DIR=./vllm_traces
  python example_vllm_internal_torch_profile.py
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

from llm_inference_profiler.trace_analyzer import TraceAnalyzer


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_MODEL = os.getenv("VLLM_MODEL", "")
VLLM_TRACE_DIR = os.getenv("VLLM_TRACE_DIR", "./vllm_traces")
PROMPT = os.getenv("PROMPT", "Explain transformers in one concise sentence.")
NUM_REQUESTS = int(os.getenv("NUM_REQUESTS", "1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "64"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "vllm_internal_torch_profile_output.json")


def _http_post_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 300) -> Dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = Request(url, method="POST", data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _http_get_json(url: str, timeout: int = 300) -> Dict[str, Any]:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _resolve_model_name() -> str:
    if VLLM_MODEL:
        return VLLM_MODEL
    models = _http_get_json(f"{VLLM_BASE_URL}/v1/models")
    data = models.get("data", [])
    if not data:
        raise RuntimeError("No models found at /v1/models")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError("Could not resolve model id from /v1/models")
    return model_id


def _send_workload(model_name: str) -> Dict[str, Any]:
    durations_ms: List[float] = []
    usages: List[Dict[str, Any]] = []

    for _ in range(NUM_REQUESTS):
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stream": False,
        }
        start = time.perf_counter()
        result = _http_post_json(f"{VLLM_BASE_URL}/v1/chat/completions", payload=payload, timeout=600)
        end = time.perf_counter()

        durations_ms.append((end - start) * 1000.0)
        if result.get("usage"):
            usages.append(result["usage"])

    aggregate = {
        "request_count": NUM_REQUESTS,
        "latency_ms": {
            "mean": round(sum(durations_ms) / len(durations_ms), 3) if durations_ms else 0.0,
            "min": round(min(durations_ms), 3) if durations_ms else 0.0,
            "max": round(max(durations_ms), 3) if durations_ms else 0.0,
        },
        "usage": usages,
    }
    return aggregate


def _latest_trace_file(trace_dir: str) -> str:
    candidates = glob.glob(os.path.join(trace_dir, "**", "*.json"), recursive=True)
    candidates += glob.glob(os.path.join(trace_dir, "**", "*.json.gz"), recursive=True)
    if not candidates:
        raise RuntimeError(f"No trace files found in {trace_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _materialize_trace(trace_path: str) -> str:
    if trace_path.endswith(".json"):
        return trace_path

    if not trace_path.endswith(".json.gz"):
        raise RuntimeError(f"Unsupported trace file format: {trace_path}")

    fd, temp_json = tempfile.mkstemp(prefix="vllm_trace_", suffix=".json")
    os.close(fd)
    with gzip.open(trace_path, "rb") as src, open(temp_json, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return temp_json


def _build_output(model_name: str, workload: Dict[str, Any], trace_results: Dict[str, Any], raw_trace_path: str) -> Dict[str, Any]:
    usages = workload.get("usage", [])
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for u in usages:
        total_prompt_tokens += int(u.get("prompt_tokens") or 0)
        total_completion_tokens += int(u.get("completion_tokens") or 0)

    mean_latency = workload.get("latency_ms", {}).get("mean", 0.0)
    tokens_per_second = 0.0
    if mean_latency > 0 and total_completion_tokens > 0 and workload.get("request_count", 0) > 0:
        avg_completion_tokens = total_completion_tokens / workload["request_count"]
        tokens_per_second = round(avg_completion_tokens / (mean_latency / 1000.0), 2)

    return {
        "metadata": {
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "backend": {
                "backend": "vllm_internal_torch_profiler",
                "provider": "vllm",
                "base_url": VLLM_BASE_URL,
                "trace_file": raw_trace_path,
                "trace_dir": VLLM_TRACE_DIR,
            },
        },
        "summary": {
            "latency": {
                "mean_request_latency_ms": mean_latency,
                "tokens_per_second": tokens_per_second,
                "request_count": workload.get("request_count", 0),
                "input_tokens_total": total_prompt_tokens,
                "output_tokens_total": total_completion_tokens,
            }
        },
        "diagnostics": {
            "trace_event_categories": trace_results.get("event_categories", {}),
            "workload": workload,
        },
        "drilldown": {
            "cuda_kernels": trace_results.get("cuda_kernels", []),
            "cpu_gpu_gaps": trace_results.get("cpu_gpu_gaps", {}),
            "gpu_memcpy": trace_results.get("gpu_memcpy", []),
        },
    }


def main() -> None:
    model_name = _resolve_model_name()
    print(f"Using model: {model_name}")

    print("Starting vLLM internal profiling window...")
    _http_post_json(f"{VLLM_BASE_URL}/start_profile")

    try:
        workload = _send_workload(model_name)
    finally:
        print("Stopping vLLM internal profiling window...")
        _http_post_json(f"{VLLM_BASE_URL}/stop_profile")

    raw_trace_path = _latest_trace_file(VLLM_TRACE_DIR)
    temp_trace: Optional[str] = None

    try:
        trace_path = _materialize_trace(raw_trace_path)
        if trace_path != raw_trace_path:
            temp_trace = trace_path

        analyzer = TraceAnalyzer(trace_path)
        trace_results = analyzer.analyze()

        output = _build_output(model_name, workload, trace_results, raw_trace_path)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"Saved: {OUTPUT_FILE}")
        print(f"Trace source: {raw_trace_path}")
    finally:
        if temp_trace and os.path.exists(temp_trace):
            try:
                os.unlink(temp_trace)
            except OSError:
                pass


if __name__ == "__main__":
    main()
