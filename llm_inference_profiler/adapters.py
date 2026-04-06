"""
Adapter interfaces for profiling different inference backends.

This module introduces a backend abstraction so LLMProfiler can profile:
- In-process HuggingFace models (existing path)
- OpenAI-compatible HTTP servers like vLLM and Ollama
- Future providers via additional adapters
"""

from __future__ import annotations

import json
import time
import os
import tempfile
import statistics
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

import torch
from torch.profiler import profile, ProfilerActivity, record_function

from .memory_tracker import MemoryTracker
from .event_analyzer import EventAnalyzer
from .trace_analyzer import TraceAnalyzer
from .utils import get_gpu_info


class BaseInferenceAdapter(ABC):
    """Base interface for remote inference adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Static metadata for this backend/model."""

    @abstractmethod
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        """
        Run generation and return normalized results expected by LLMProfiler.

        Required keys in returned dict:
        - generated_text: str
        - token_results: Dict[str, Any]
        - input_token_count: Optional[int]
        - output_token_count: Optional[int]
        - diagnostics: Dict[str, Any]
        - drilldown: Dict[str, Any]
        """


class OpenAICompatibleAdapter(BaseInferenceAdapter):
    """
    Adapter for OpenAI-compatible chat APIs (vLLM, Ollama, etc.).

    Notes:
    - Uses streaming responses to estimate TTFT and ITL.
    - ITL is chunk-based because API streaming may emit partial token chunks.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "",
        provider: str = "openai-compatible",
        timeout_sec: int = 300,
        collect_metrics: bool = True,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._provider = provider
        self._timeout_sec = timeout_sec
        self._collect_metrics = collect_metrics
        self._extra_headers = extra_headers or {}

    @property
    def name(self) -> str:
        return self._provider

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "remote_api",
            "adapter": "OpenAICompatibleAdapter",
            "provider": self._provider,
            "base_url": self._base_url,
            "model_name": self._resolve_model_name(),
        }

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        if messages is None and prompt is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        model_name = self._resolve_model_name()

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if do_sample:
            payload["temperature"] = temperature
            payload["top_p"] = top_p
            if top_k and top_k > 0:
                payload["top_k"] = top_k
        else:
            payload["temperature"] = 0.0

        payload.update(generate_kwargs)

        req = Request(
            url=f"{self._base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._json_headers(),
            method="POST",
        )

        start = time.perf_counter()
        first_chunk_ts: Optional[float] = None
        stream_timestamps: List[float] = []
        completion_chunks: List[str] = []
        usage: Optional[Dict[str, Any]] = None

        with urlopen(req, timeout=self._timeout_sec) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break

                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choice = (evt.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                chunk_text = delta.get("content", "")

                if chunk_text:
                    now = time.perf_counter()
                    if first_chunk_ts is None:
                        first_chunk_ts = now
                    stream_timestamps.append(now)
                    completion_chunks.append(chunk_text)

                if evt.get("usage"):
                    usage = evt["usage"]

        end = time.perf_counter()

        generated_text = "".join(completion_chunks)

        ttft_ms = (first_chunk_ts - start) * 1000 if first_chunk_ts is not None else 0.0

        itl_values_ms: List[float] = []
        for i in range(1, len(stream_timestamps)):
            itl_values_ms.append((stream_timestamps[i] - stream_timestamps[i - 1]) * 1000)

        decode_time_ms = 0.0
        if first_chunk_ts is not None and stream_timestamps:
            decode_time_ms = (stream_timestamps[-1] - first_chunk_ts) * 1000

        total_generation_time_ms = (end - start) * 1000

        input_tokens = usage.get("prompt_tokens") if usage else None
        output_tokens = usage.get("completion_tokens") if usage else None

        if input_tokens is None:
            input_tokens = self._try_tokenize_count(model_name, messages[-1].get("content", ""))
        if output_tokens is None:
            output_tokens = self._try_tokenize_count(model_name, generated_text)

        decode_tokens = max((output_tokens or 0) - 1, 0)
        decode_tps = 0.0
        if decode_time_ms > 0 and decode_tokens > 0:
            decode_tps = round(decode_tokens / (decode_time_ms / 1000.0), 2)

        token_timeline = []
        if first_chunk_ts is not None:
            token_timeline.append(
                {
                    "token_index": 0,
                    "phase": "prefill",
                    "latency_ms": round(ttft_ms, 3),
                }
            )
            for i, itl_ms in enumerate(itl_values_ms, start=1):
                token_timeline.append(
                    {
                        "token_index": i,
                        "phase": "decode",
                        "latency_ms": round(itl_ms, 3),
                    }
                )

        token_results = {
            "num_tokens_generated": len(stream_timestamps),
            "time_to_first_token_ms": round(ttft_ms, 3),
            "prefill_time_ms": round(ttft_ms, 3),
            "decode_time_ms": round(decode_time_ms, 3),
            "total_generation_time_ms": round(total_generation_time_ms, 3),
            "tokens_per_second": decode_tps,
            "inter_token_latency": self._percentiles(itl_values_ms),
            "token_timeline": token_timeline,
        }

        diagnostics: Dict[str, Any] = {
            "remote_stream": {
                "stream_chunks": len(completion_chunks),
                "timeline_ms": [round((t - start) * 1000, 3) for t in stream_timestamps],
                "usage": usage or {},
            }
        }

        drilldown: Dict[str, Any] = {}
        if self._collect_metrics:
            drilldown["server_metrics"] = self._try_collect_metrics()

        return {
            "generated_text": generated_text,
            "token_results": token_results,
            "input_token_count": input_tokens,
            "output_token_count": output_tokens,
            "diagnostics": diagnostics,
            "drilldown": drilldown,
        }

    def _json_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self._extra_headers)
        return headers

    def _http_json(self, method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = Request(
            url=url,
            data=data,
            headers=self._json_headers(),
            method=method,
        )
        with urlopen(req, timeout=self._timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_text(self, method: str, url: str) -> str:
        req = Request(url=url, method=method, headers=self._extra_headers)
        with urlopen(req, timeout=self._timeout_sec) as resp:
            return resp.read().decode("utf-8")

    def _resolve_model_name(self) -> str:
        if self._model_name:
            return self._model_name

        models = self._http_json("GET", f"{self._base_url}/v1/models")
        data = models.get("data", [])
        if not data:
            raise RuntimeError("No models returned by /v1/models")

        model_id = data[0].get("id")
        if not model_id:
            raise RuntimeError("Could not infer model id from /v1/models")

        self._model_name = model_id
        return self._model_name

    def _try_tokenize_count(self, model: str, text: str) -> Optional[int]:
        if not text:
            return 0

        attempts = [
            {"model": model, "prompt": text},
            {"model": model, "text": text},
        ]

        for payload in attempts:
            try:
                out = self._http_json("POST", f"{self._base_url}/tokenize", payload)
                token_ids = out.get("token_ids")
                if isinstance(token_ids, list):
                    return len(token_ids)
                tokens = out.get("tokens")
                if isinstance(tokens, list):
                    return len(tokens)
            except Exception:
                continue

        return None

    def _try_collect_metrics(self) -> Dict[str, Any]:
        try:
            metrics_text = self._http_text("GET", f"{self._base_url}/metrics")
        except Exception as exc:
            return {
                "available": False,
                "error": str(exc),
            }

        interesting_prefixes = (
            "vllm:",
            "vllm_",
            "ollama_",
            "process_",
            "python_gc_",
        )

        selected = []
        for line in metrics_text.splitlines():
            if not line or line.startswith("#"):
                continue
            if line.startswith(interesting_prefixes):
                selected.append(line)

        return {
            "available": True,
            "metrics_line_count": len(metrics_text.splitlines()),
            "selected_metrics": selected,
        }

    @staticmethod
    def _percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}

        s = sorted(values)

        def q(p: float) -> float:
            if len(s) == 1:
                return s[0]
            idx = (len(s) - 1) * p
            low = int(idx)
            high = min(low + 1, len(s) - 1)
            frac = idx - low
            return s[low] * (1 - frac) + s[high] * frac

        return {
            "mean_ms": round(statistics.fmean(values), 3),
            "median_ms": round(statistics.median(values), 3),
            "p90_ms": round(q(0.90), 3),
            "p95_ms": round(q(0.95), 3),
            "p99_ms": round(q(0.99), 3),
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
        }


class LocalVLLMAdapter(BaseInferenceAdapter):
    """
    In-process vLLM adapter.

    This keeps execution and profiling in the same machine/process context,
    enabling deep operator/kernel visibility through torch.profiler.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        top_n: int = 20,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        vllm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vllm is required for LocalVLLMAdapter. Install with: pip install vllm"
            ) from exc

        self._model_name = model_name
        self._top_n = top_n
        self._record_shapes = record_shapes
        self._profile_memory = profile_memory
        self._with_stack = with_stack
        self._with_flops = with_flops
        self._has_cuda = torch.cuda.is_available()

        kwargs = dict(vllm_kwargs or {})
        kwargs.setdefault("model", model_name)
        kwargs.setdefault("tensor_parallel_size", tensor_parallel_size)
        kwargs.setdefault("trust_remote_code", trust_remote_code)
        kwargs.setdefault("dtype", dtype)
        kwargs.setdefault("gpu_memory_utilization", gpu_memory_utilization)

        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        self._memory_tracker = MemoryTracker()
        self._gpu_info = get_gpu_info()

    @property
    def name(self) -> str:
        return "local-vllm"

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "local_vllm",
            "adapter": "LocalVLLMAdapter",
            "provider": "vllm",
            "model_name": self._model_name,
            "gpu": self._gpu_info,
        }

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vllm is required for LocalVLLMAdapter. Install with: pip install vllm"
            ) from exc

        if messages is None and prompt is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        tokenization_start = time.perf_counter()
        if messages is not None:
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = "\n".join(m.get("content", "") for m in messages)
        else:
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt_text = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt or ""}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = prompt or ""

        input_ids = self._tokenizer(prompt_text, return_tensors="pt").get("input_ids")
        input_token_count = int(input_ids.shape[1]) if input_ids is not None else 0
        tokenization_time_ms = (time.perf_counter() - tokenization_start) * 1000

        sampling_kwargs: Dict[str, Any] = {
            "max_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
        }
        if top_k > 0:
            sampling_kwargs["top_k"] = top_k
        sampling_kwargs.update(generate_kwargs)
        sampling_params = SamplingParams(**sampling_kwargs)

        if self._has_cuda:
            self._memory_tracker.reset_peak()
            torch.cuda.empty_cache()
        self._memory_tracker.snapshot("before_inference")

        activities = [ProfilerActivity.CPU]
        if self._has_cuda:
            activities.append(ProfilerActivity.CUDA)

        trace_dir = tempfile.mkdtemp(prefix="llm_profiler_vllm_")
        trace_path = os.path.join(trace_dir, "trace.json")

        start = time.perf_counter()
        with profile(
            activities=activities,
            record_shapes=self._record_shapes,
            profile_memory=self._profile_memory,
            with_stack=self._with_stack,
            with_flops=self._with_flops,
        ) as prof:
            with record_function("VLLM_GENERATE"):
                outputs = self._llm.generate([prompt_text], sampling_params, use_tqdm=False)
        end = time.perf_counter()

        self._memory_tracker.snapshot("after_prefill")
        self._memory_tracker.snapshot("after_inference")

        prof.export_chrome_trace(trace_path)

        out = outputs[0]
        generated = out.outputs[0] if out.outputs else None
        generated_text = generated.text if generated is not None else ""
        output_token_count = len(generated.token_ids) if generated is not None and generated.token_ids is not None else 0

        total_generation_time_ms = (end - start) * 1000
        metrics = getattr(out, "metrics", None)

        ttft_ms = 0.0
        decode_time_ms = 0.0
        itl_values_ms: List[float] = []
        if metrics is not None:
            first_token_time = getattr(metrics, "first_token_time", None)
            arrival_time = getattr(metrics, "arrival_time", None)
            finished_time = getattr(metrics, "finished_time", None)
            if first_token_time is not None and arrival_time is not None:
                ttft_ms = (first_token_time - arrival_time) * 1000
            if first_token_time is not None and finished_time is not None:
                decode_time_ms = max((finished_time - first_token_time) * 1000, 0.0)

        if decode_time_ms <= 0:
            decode_time_ms = max(total_generation_time_ms - ttft_ms, 0.0)

        decode_tokens = max(output_token_count - 1, 0)
        decode_tps = 0.0
        if decode_time_ms > 0 and decode_tokens > 0:
            decode_tps = round(decode_tokens / (decode_time_ms / 1000.0), 2)

        token_results = {
            "num_tokens_generated": output_token_count,
            "time_to_first_token_ms": round(ttft_ms, 3),
            "prefill_time_ms": round(ttft_ms, 3),
            "decode_time_ms": round(decode_time_ms, 3),
            "total_generation_time_ms": round(total_generation_time_ms, 3),
            "tokens_per_second": decode_tps,
            "inter_token_latency": self._percentiles(itl_values_ms),
            "token_timeline": [],
        }

        gpu_name = self._gpu_info.get("name", "") if self._gpu_info.get("available") else ""
        event_analyzer = EventAnalyzer(
            prof=prof,
            gpu_name=gpu_name,
            model_dtype=torch.float16,
            top_n=self._top_n,
        )
        event_results = event_analyzer.analyze()

        trace_analyzer = TraceAnalyzer(trace_path)
        trace_results = trace_analyzer.analyze()

        memory_summary = self._memory_tracker.get_summary()

        try:
            os.unlink(trace_path)
            os.rmdir(trace_dir)
        except OSError:
            pass

        return {
            "generated_text": generated_text,
            "token_results": token_results,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "tokenization_time_ms": tokenization_time_ms,
            "detokenization_time_ms": 0.0,
            "memory_summary": memory_summary,
            "layer_breakdown": {"summary": {}, "per_layer": []},
            "event_results": event_results,
            "trace_results": trace_results,
            "diagnostics": {
                "vllm": {
                    "request_id": getattr(out, "request_id", None),
                    "metrics": metrics.__dict__ if hasattr(metrics, "__dict__") else {},
                }
            },
            "drilldown": {},
        }

    @staticmethod
    def _percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}

        s = sorted(values)

        def q(p: float) -> float:
            if len(s) == 1:
                return s[0]
            idx = (len(s) - 1) * p
            low = int(idx)
            high = min(low + 1, len(s) - 1)
            frac = idx - low
            return s[low] * (1 - frac) + s[high] * frac

        return {
            "mean_ms": round(statistics.fmean(values), 3),
            "median_ms": round(statistics.median(values), 3),
            "p90_ms": round(q(0.90), 3),
            "p95_ms": round(q(0.95), 3),
            "p99_ms": round(q(0.99), 3),
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
        }
