"""
Profiled local vLLM server wrapper.

This server runs vLLM and profiler in the same process/machine.
For every incoming chat request, it:
1) executes generation through LocalVLLMAdapter,
2) captures deep profiling data (torch profiler + trace analysis),
3) saves per-request profile JSON,
4) returns OpenAI-compatible response.

Run:
  export MODEL_NAME=Qwen/Qwen3-4B
  export HOST=0.0.0.0
  export PORT=9000
  python profiled_vllm_server.py
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from llm_inference_profiler import LLMProfiler, LocalVLLMAdapter


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9000"))
PROFILE_OUTPUT_DIR = Path(os.getenv("PROFILE_OUTPUT_DIR", "profiled_requests"))

TEMPERATURE_DEFAULT = float(os.getenv("TEMPERATURE", "0.0"))
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS", "128"))
TOP_P_DEFAULT = float(os.getenv("TOP_P", "1.0"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "50"))

TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    stream: Optional[bool] = Field(default=False)


PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

adapter = LocalVLLMAdapter(
    model_name=MODEL_NAME,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
    with_flops=True,
)
profiler = LLMProfiler(adapter=adapter)

app = FastAPI(title="Profiled vLLM Wrapper", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest) -> JSONResponse:
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true not supported in profiled wrapper yet")

    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    temperature = TEMPERATURE_DEFAULT if req.temperature is None else req.temperature
    max_new_tokens = MAX_NEW_TOKENS_DEFAULT if req.max_tokens is None else req.max_tokens
    top_p = TOP_P_DEFAULT if req.top_p is None else req.top_p
    top_k = TOP_K_DEFAULT if req.top_k is None else req.top_k

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    start = time.perf_counter()
    try:
        output_text = profiler.generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    elapsed = time.perf_counter() - start

    results = profiler.results() or {}
    profile_path = PROFILE_OUTPUT_DIR / f"{request_id}.json"
    profiler.save(str(profile_path))

    latency = (results.get("summary") or {}).get("latency") or {}
    prompt_tokens = int(latency.get("input_tokens") or 0)
    completion_tokens = int(latency.get("output_tokens") or 0)

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "profiling": {
            "enabled": True,
            "request_wall_time_ms": round(elapsed * 1000.0, 3),
            "profile_json": str(profile_path),
        },
    }

    return JSONResponse(content=response)


@app.get("/profiles/{request_id}")
def get_profile(request_id: str) -> JSONResponse:
    profile_path = PROFILE_OUTPUT_DIR / f"{request_id}.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")

    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


if __name__ == "__main__":
    print(f"Starting profiled wrapper on http://{HOST}:{PORT}")
    print(f"Model: {MODEL_NAME}")
    print(f"Profile output dir: {PROFILE_OUTPUT_DIR}")
    uvicorn.run(app, host=HOST, port=PORT)
