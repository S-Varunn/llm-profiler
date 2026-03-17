"""
Main LLM Inference Profiler.
Orchestrates all sub-components to produce a consolidated JSON profile
of an LLM inference run.
"""

import time
import os
import tempfile
import datetime
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from typing import Any, Dict, List, Optional, Union

from .hooks import LayerHooks
from .token_tracker import TokenTimingProcessor
from .memory_tracker import MemoryTracker
from .event_analyzer import EventAnalyzer
from .trace_analyzer import TraceAnalyzer
from .utils import (
    get_gpu_info,
    get_model_info,
    save_json,
    safe_json_serialize,
    us_to_ms,
    bytes_to_mb,
)


class LLMProfiler:
    """
    Passive profiler for LLM inference.

    Wraps model.generate() with:
    - PyTorch Profiler (operator/kernel level)
    - Module hooks (layer-level CUDA event timing)
    - LogitsProcessor (per-token TTFT/ITL)
    - Memory snapshots

    Does not modify model outputs. Outputs a consolidated JSON.

    Usage:
        profiler = LLMProfiler(model=model, tokenizer=tokenizer)
        output = profiler.generate(
            prompt="What is ML?",
            max_new_tokens=50,
        )
        profiler.save("profile.json")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        top_n: int = 20,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        with_modules: bool = False,
    ):
        """
        Args:
            model: HuggingFace causal LM model.
            tokenizer: HuggingFace tokenizer.
            top_n: Number of top items to include in breakdowns.
            record_shapes: Record tensor input shapes (moderate memory cost).
            profile_memory: Track per-operator memory allocations.
            with_stack: Capture Python stack traces (HIGH memory cost — enable
                        only if you need drilldown.top_expensive_ops[].stack_trace).
            with_flops: Estimate FLOPS for supported operators.
            with_modules: Associate operators with nn.Module hierarchy.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._top_n = top_n
        self._has_cuda = torch.cuda.is_available()
        self._results: Optional[Dict[str, Any]] = None

        # Profiler options
        self._record_shapes = record_shapes
        self._profile_memory = profile_memory
        self._with_stack = with_stack
        self._with_flops = with_flops
        self._with_modules = with_modules

        # Sub-components
        self._token_tracker = TokenTimingProcessor()
        self._memory_tracker = MemoryTracker()
        self._layer_hooks: Optional[LayerHooks] = None

        # Model/GPU metadata (collected once)
        self._gpu_info = get_gpu_info()
        self._model_info = get_model_info(model)

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
    ) -> str:
        """
        Run model.generate() with profiling enabled.

        Args:
            prompt: Raw text prompt. Either prompt or messages must be provided.
            messages: Chat messages list [{"role": "user", "content": "..."}].
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            **generate_kwargs: Additional kwargs passed to model.generate().

        Returns:
            Generated text string (identical to calling model.generate directly).
        """
        # Reset state
        self._results = None
        self._token_tracker.reset()

        # =====================================================================
        # Phase 0: Tokenization
        # =====================================================================
        tokenization_start = time.perf_counter()

        if messages is not None:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif prompt is not None:
            if hasattr(self._tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = prompt
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(text, return_tensors="pt").to(device)
        input_token_count = inputs["input_ids"].shape[1]

        if self._has_cuda:
            torch.cuda.synchronize()
        tokenization_time_ms = (time.perf_counter() - tokenization_start) * 1000

        # =====================================================================
        # Phase 1: Setup profiling
        # =====================================================================

        # Attach layer hooks
        self._layer_hooks = LayerHooks(self._model)

        # Memory baseline
        if self._has_cuda:
            self._memory_tracker.reset_peak()
        self._memory_tracker.snapshot("before_inference")

        # Build generate kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
            "logits_processor": [self._token_tracker],
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k
        gen_kwargs.update(generate_kwargs)

        # =====================================================================
        # Phase 2: Run generation with PyTorch Profiler
        # =====================================================================

        activities = [ProfilerActivity.CPU]
        if self._has_cuda:
            activities.append(ProfilerActivity.CUDA)
            # Free cached memory to give profiler room
            torch.cuda.empty_cache()

        # Temp file for chrome trace
        trace_dir = tempfile.mkdtemp(prefix="llm_profiler_")
        trace_path = os.path.join(trace_dir, "trace.json")

        self._token_tracker.mark_start()

        with profile(
            activities=activities,
            record_shapes=self._record_shapes,
            profile_memory=self._profile_memory,
            with_stack=self._with_stack,
            with_flops=self._with_flops,
            with_modules=self._with_modules,
        ) as prof:
            with record_function("LLM_GENERATE"):
                with torch.no_grad():
                    outputs = self._model.generate(**inputs, **gen_kwargs)

        # Post-prefill memory snapshot (approximate — taken after full generation)
        # We use token_tracker timestamps to identify the prefill boundary
        self._memory_tracker.snapshot("after_prefill")

        # Final memory snapshot
        self._memory_tracker.snapshot("after_inference")

        # Export chrome trace for TraceAnalyzer
        prof.export_chrome_trace(trace_path)

        # =====================================================================
        # Phase 3: Detokenization
        # =====================================================================
        detok_start = time.perf_counter()
        generated_text = self._tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        if self._has_cuda:
            torch.cuda.synchronize()
        detokenization_time_ms = (time.perf_counter() - detok_start) * 1000

        output_token_count = outputs.shape[1] - input_token_count

        # =====================================================================
        # Phase 4: Collect and assemble results
        # =====================================================================

        # Layer hook results
        layer_breakdown = self._layer_hooks.compute_results()
        self._layer_hooks.remove_hooks()

        # Token timing results
        token_results = self._token_tracker.get_results()

        # Memory results
        memory_summary = self._memory_tracker.get_summary()

        # Event analysis (PyTorch profiler events)
        gpu_name = self._gpu_info.get("name", "") if self._gpu_info.get("available") else ""
        dtype_str = self._model_info["dtype"].replace("torch.", "")
        model_dtype = getattr(torch, dtype_str, torch.float16)

        event_analyzer = EventAnalyzer(
            prof=prof,
            gpu_name=gpu_name,
            model_dtype=model_dtype,
            top_n=self._top_n,
        )
        event_results = event_analyzer.analyze()

        # Trace analysis (chrome trace JSON)
        trace_analyzer = TraceAnalyzer(trace_path)
        trace_results = trace_analyzer.analyze()

        # =====================================================================
        # Phase 5: Assemble final JSON
        # =====================================================================

        self._results = self._assemble_json(
            tokenization_time_ms=tokenization_time_ms,
            detokenization_time_ms=detokenization_time_ms,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            token_results=token_results,
            memory_summary=memory_summary,
            layer_breakdown=layer_breakdown,
            event_results=event_results,
            trace_results=trace_results,
            trace_path=trace_path,
        )

        # Cleanup temp trace file
        try:
            os.unlink(trace_path)
            os.rmdir(trace_dir)
        except OSError:
            pass

        return generated_text

    def _assemble_json(
        self,
        tokenization_time_ms: float,
        detokenization_time_ms: float,
        input_token_count: int,
        output_token_count: int,
        token_results: Dict[str, Any],
        memory_summary: Dict[str, Any],
        layer_breakdown: Dict[str, Any],
        event_results: Dict[str, Any],
        trace_results: Dict[str, Any],
        trace_path: str,
        profiler_failed: bool = False,
    ) -> Dict[str, Any]:
        """Assemble the three-layer JSON output."""

        # Extract timing values from token tracker
        ttft_ms = token_results.get("time_to_first_token_ms", 0)
        decode_ms = token_results.get("decode_time_ms", 0)
        total_gen_ms = token_results.get("total_generation_time_ms", 0)
        tps = token_results.get("tokens_per_second", 0)
        itl = token_results.get("inter_token_latency", {})

        # Total wall time includes tokenization + generation + detokenization
        total_wall_ms = tokenization_time_ms + total_gen_ms + detokenization_time_ms

        return {
            # =================================================================
            # Metadata
            # =================================================================
            "metadata": {
                "model_name": getattr(self._model, 'name_or_path',
                                      getattr(self._model.config, '_name_or_path', 'unknown'))
                    if hasattr(self._model, 'config') else 'unknown',
                "device": self._model_info["device"],
                "gpu": self._gpu_info,
                "dtype": self._model_info["dtype"],
                "num_parameters": self._model_info["num_parameters"],
                "num_parameters_human": self._model_info["num_parameters_human"],
                "num_layers": self._model_info["num_layers"],
                "timestamp": datetime.datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if self._has_cuda else None,
                "profiler_settings": {
                    "record_shapes": self._record_shapes,
                    "profile_memory": self._profile_memory,
                    "with_stack": self._with_stack,
                    "with_flops": self._with_flops,
                    "with_modules": self._with_modules,
                    "profiler_fallback": profiler_failed,
                },
            },

            # =================================================================
            # Summary (Layer 1 — quick glance)
            # =================================================================
            "summary": {
                "latency": {
                    "total_wall_time_ms": round(total_wall_ms, 3),
                    "tokenization_ms": round(tokenization_time_ms, 3),
                    "prefill_ms": round(ttft_ms, 3),
                    "decode_ms": round(decode_ms, 3),
                    "detokenization_ms": round(detokenization_time_ms, 3),
                    "time_to_first_token_ms": round(ttft_ms, 3),
                    "tokens_per_second": tps,
                    "inter_token_latency": itl,
                    "input_tokens": input_token_count,
                    "output_tokens": output_token_count,
                },
                "memory": memory_summary,
                "layer_breakdown": layer_breakdown.get("summary", {}),
            },

            # =================================================================
            # Diagnostics (Layer 2 — find the bottleneck)
            # =================================================================
            "diagnostics": {
                **event_results.get("diagnostics", {}),
                "per_layer": layer_breakdown.get("per_layer", []),
                "token_timeline": token_results.get("token_timeline", []),
                "trace_event_categories": trace_results.get(
                    "event_categories", {}
                ),
            },

            # =================================================================
            # Drilldown (Layer 3 — understand the bottleneck)
            # =================================================================
            "drilldown": {
                **event_results.get("drilldown", {}),
                "cuda_kernels": trace_results.get("cuda_kernels", []),
                "cpu_gpu_gaps": trace_results.get("cpu_gpu_gaps", {}),
                "gpu_memcpy": trace_results.get("gpu_memcpy", []),
                "memory_snapshots": self._memory_tracker.get_snapshots(),
            },
        }

    def results(self) -> Optional[Dict[str, Any]]:
        """Return the profiling results dict. None if generate() hasn't been called."""
        return self._results

    def save(self, filepath: str, indent: int = 2) -> None:
        """Save profiling results to a JSON file."""
        if self._results is None:
            raise RuntimeError(
                "No results to save. Call generate() first."
            )
        save_json(self._results, filepath, indent=indent)
        print(f"Profile saved to: {filepath}")

    def summary(self) -> None:
        """Print a human-readable summary of the profiling results."""
        if self._results is None:
            print("No results. Call generate() first.")
            return

        r = self._results
        s = r["summary"]
        lat = s["latency"]
        mem = s["memory"]
        lb = s["layer_breakdown"]

        print("=" * 60)
        print("LLM INFERENCE PROFILE SUMMARY")
        print("=" * 60)
        print(f"Model: {r['metadata'].get('model_name', 'unknown')}")
        print(f"Device: {r['metadata']['device']}")
        print(f"Input tokens: {lat['input_tokens']}")
        print(f"Output tokens: {lat['output_tokens']}")
        print()

        print("--- Latency ---")
        print(f"  Total wall time:       {lat['total_wall_time_ms']:.1f} ms")
        print(f"  Tokenization:          {lat['tokenization_ms']:.1f} ms")
        print(f"  Prefill (TTFT):        {lat['prefill_ms']:.1f} ms")
        print(f"  Decode:                {lat['decode_ms']:.1f} ms")
        print(f"  Detokenization:        {lat['detokenization_ms']:.1f} ms")
        print(f"  Tokens/sec:            {lat['tokens_per_second']:.1f}")

        itl = lat.get("inter_token_latency", {})
        if itl:
            print(f"  ITL mean:              {itl.get('mean_ms', 0):.1f} ms")
            print(f"  ITL p90:               {itl.get('p90_ms', 0):.1f} ms")
            print(f"  ITL p99:               {itl.get('p99_ms', 0):.1f} ms")

        print()
        print("--- Memory ---")
        print(f"  Before inference:      {mem.get('before_inference_mb', 0):.0f} MB")
        print(f"  Peak:                  {mem.get('peak_during_inference_mb', 0):.0f} MB")
        print(f"  Delta:                 {mem.get('inference_delta_mb', 0):.0f} MB")
        print(f"  GPU utilization:       {mem.get('gpu_utilization_pct', 0):.1f}%")

        print()
        print("--- Layer Breakdown ---")
        if lb:
            print(f"  Attention:             {lb.get('attention_pct', 0):.1f}%")
            print(f"  MLP:                   {lb.get('mlp_pct', 0):.1f}%")
            print(f"  Normalization:         {lb.get('normalization_pct', 0):.1f}%")
            print(f"  Embedding:             {lb.get('embedding_pct', 0):.1f}%")
            print(f"  LM Head:               {lb.get('lm_head_pct', 0):.1f}%")

        print("=" * 60)
