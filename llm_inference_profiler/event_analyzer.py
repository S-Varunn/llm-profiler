"""
Event analyzer for PyTorch Profiler output.
Processes prof.events() and prof.key_averages() to produce:
  - Operator breakdown (top ops by GPU time, CPU time)
  - Per-phase analysis (prefill vs decode by input shape)
  - Stack traces for top expensive operations
  - FLOPS and roofline analysis
  - Memory allocation hotspots
  - Sync point detection
"""

import torch
from typing import Any, Dict, List, Optional
from .utils import us_to_ms, bytes_to_mb, compute_roofline


class EventAnalyzer:
    """
    Analyzes PyTorch profiler events to build the diagnostics and drilldown
    JSON sections.
    """

    def __init__(
        self,
        prof: torch.profiler.profile,
        gpu_name: str = "",
        model_dtype: torch.dtype = torch.float16,
        top_n: int = 20,
    ):
        self._prof = prof
        self._gpu_name = gpu_name
        self._model_dtype = model_dtype
        self._top_n = top_n
        self._has_cuda = torch.cuda.is_available()

    def analyze(self) -> Dict[str, Any]:
        """
        Run full analysis and return diagnostics + drilldown sections.
        """
        diagnostics = self._build_diagnostics()
        drilldown = self._build_drilldown()
        return {
            "diagnostics": diagnostics,
            "drilldown": drilldown,
        }

    # -------------------------------------------------------------------------
    # Diagnostics Section
    # -------------------------------------------------------------------------

    def _build_diagnostics(self) -> Dict[str, Any]:
        """Build the diagnostics section with operator/kernel breakdowns."""

        # --- Top operators (aggregated) ---
        top_ops_gpu = self._get_top_operators(sort_by="gpu")
        top_ops_cpu = self._get_top_operators(sort_by="cpu")

        # --- Prefill vs decode operators (by input shape) ---
        prefill_ops, decode_ops = self._get_phase_operators()

        # --- Top CUDA kernels ---
        top_kernels = self._get_top_cuda_kernels()

        # --- Sync points ---
        sync_points = self._get_sync_points()

        # --- Memory operations ---
        mem_ops = self._get_memory_operations()

        return {
            "top_operators_by_gpu_time": top_ops_gpu,
            "top_operators_by_cpu_time": top_ops_cpu,
            "prefill_operators": prefill_ops,
            "decode_operators": decode_ops,
            "top_cuda_kernels": top_kernels,
            "sync_points": sync_points,
            "memory_operations": mem_ops,
        }

    def _get_top_operators(self, sort_by: str = "gpu") -> List[Dict[str, Any]]:
        """Get top N operators sorted by GPU or CPU time."""
        events = self._prof.key_averages()

        if sort_by == "gpu":
            events_sorted = sorted(
                events, key=lambda e: e.self_device_time_total, reverse=True
            )
        else:
            events_sorted = sorted(
                events, key=lambda e: e.self_cpu_time_total, reverse=True
            )

        results = []
        for evt in events_sorted[: self._top_n]:
            entry = {
                "name": evt.key,
                "calls": evt.count,
                "cpu_time_total_ms": us_to_ms(evt.cpu_time_total),
                "self_cpu_time_ms": us_to_ms(evt.self_cpu_time_total),
                "gpu_time_total_ms": us_to_ms(evt.device_time_total),
                "self_gpu_time_ms": us_to_ms(evt.self_device_time_total),
            }

            if evt.flops > 0:
                entry["flops"] = evt.flops
                entry["gflops"] = round(evt.flops / 1e9, 2)

            cpu_mem = getattr(evt, 'cpu_memory_usage', 0) or 0
            cuda_mem = getattr(evt, 'cuda_memory_usage', getattr(evt, 'self_cuda_memory_usage', 0)) or 0
            if cpu_mem != 0:
                entry["cpu_memory_mb"] = bytes_to_mb(cpu_mem)
            if cuda_mem != 0:
                entry["cuda_memory_mb"] = bytes_to_mb(cuda_mem)

            results.append(entry)

        return results

    def _get_phase_operators(
        self,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Separate prefill and decode operators by input shape.
        Prefill ops have seq_len > 1 in their input shapes.
        Decode ops have seq_len == 1.
        """
        try:
            events = self._prof.key_averages(group_by_input_shape=True)
        except Exception:
            return [], []

        prefill = []
        decode = []

        for evt in events:
            if evt.self_device_time_total <= 0 and evt.self_cpu_time_total <= 0:
                continue

            entry = {
                "name": evt.key,
                "calls": evt.count,
                "self_gpu_time_ms": us_to_ms(evt.self_device_time_total),
                "self_cpu_time_ms": us_to_ms(evt.self_cpu_time_total),
                "input_shapes": [list(s) if hasattr(s, '__iter__') else s
                                 for s in evt.input_shapes] if evt.input_shapes else [],
            }

            if evt.flops > 0:
                entry["flops"] = evt.flops

            # Classify by shape: if any dimension is > 1 in a
            # likely sequence-length position, it's prefill
            is_prefill = self._is_prefill_shape(evt.input_shapes)

            if is_prefill:
                prefill.append(entry)
            else:
                decode.append(entry)

        # Sort each by GPU time
        prefill.sort(key=lambda x: x["self_gpu_time_ms"], reverse=True)
        decode.sort(key=lambda x: x["self_gpu_time_ms"], reverse=True)

        return prefill[: self._top_n], decode[: self._top_n]

    def _is_prefill_shape(self, shapes: Any) -> bool:
        """
        Heuristic: if input shapes contain a dimension > 1 that looks like
        a sequence length (typically the second dimension in [batch, seq, ...]),
        it's likely a prefill operation.
        """
        if not shapes:
            return False

        for shape in shapes:
            if not hasattr(shape, '__len__') or len(shape) < 2:
                continue
            # For 3D tensors [batch, seq, hidden], check seq dim
            if len(shape) == 3 and shape[1] > 1:
                return True
            # For 2D tensors [seq, hidden] or [batch, seq]
            if len(shape) == 2 and shape[0] > 1 and shape[0] < 100000:
                return True

        return False

    def _get_top_cuda_kernels(self) -> List[Dict[str, Any]]:
        """Extract top CUDA kernels by GPU time."""
        events = self._prof.key_averages()

        kernels = [
            e for e in events
            if e.self_device_time_total > 0
            and not e.key.startswith("aten::")
            and not e.key.startswith("torch::")
            and not e.key.startswith("cudaLaunch")
            and not e.key.startswith("cudaMalloc")
            and not e.key.startswith("cudaFree")
            and not e.key.startswith("cudaStream")
            and not e.key.startswith("cudaMemcpy")
            and not e.key.startswith("cudaMemset")
        ]

        # If nothing left, include aten ops with device time
        if not kernels:
            kernels = [e for e in events if e.self_device_time_total > 0]

        kernels.sort(key=lambda e: e.self_device_time_total, reverse=True)

        results = []
        for evt in kernels[: self._top_n]:
            results.append({
                "name": evt.key,
                "gpu_time_ms": us_to_ms(evt.self_device_time_total),
                "calls": evt.count,
            })

        return results

    def _get_sync_points(self) -> List[Dict[str, Any]]:
        """Find CUDA synchronization events (pipeline stalls)."""
        events = self._prof.key_averages()

        sync_events = [
            e for e in events
            if any(x in e.key.lower() for x in [
                "synchronize", "cudadevicesynchronize",
                "cudastreamsynchronize", "cudaeventsynchronize",
            ])
        ]

        results = []
        for evt in sync_events:
            results.append({
                "name": evt.key,
                "count": evt.count,
                "total_stall_ms": us_to_ms(evt.cpu_time_total),
                "avg_stall_ms": us_to_ms(
                    evt.cpu_time_total / evt.count
                ) if evt.count > 0 else 0,
            })

        return results

    def _get_memory_operations(self) -> List[Dict[str, Any]]:
        """Find memory transfer and allocation events."""
        events = self._prof.key_averages()

        mem_events = [
            e for e in events
            if any(x in e.key.lower() for x in [
                "memcpy", "memset", "to_copy", "cudamalloc", "cudafree",
                "copy_", "contiguous",
            ])
        ]

        results = []
        for evt in sorted(mem_events, key=lambda e: e.cpu_time_total, reverse=True):
            results.append({
                "name": evt.key,
                "count": evt.count,
                "total_ms": us_to_ms(evt.cpu_time_total),
                "gpu_time_ms": us_to_ms(evt.self_device_time_total),
            })

        return results[: self._top_n]

    # -------------------------------------------------------------------------
    # Drilldown Section
    # -------------------------------------------------------------------------

    def _build_drilldown(self) -> Dict[str, Any]:
        """Build the drilldown section with stack traces, roofline, etc."""

        top_expensive = self._get_top_expensive_with_stacks()
        memory_allocators = self._get_memory_allocators()

        return {
            "top_expensive_ops": top_expensive,
            "memory_allocations": memory_allocators,
        }

    def _get_top_expensive_with_stacks(self) -> List[Dict[str, Any]]:
        """
        Get the top expensive operations with stack traces, roofline analysis,
        and input shape distribution.
        """
        # Use stack-grouped averages for stack traces
        try:
            stack_events = self._prof.key_averages(group_by_stack_n=5)
        except Exception:
            stack_events = self._prof.key_averages()

        # Sort by GPU time, fall back to CPU time
        if self._has_cuda:
            stack_events_sorted = sorted(
                stack_events,
                key=lambda e: e.self_device_time_total,
                reverse=True,
            )
        else:
            stack_events_sorted = sorted(
                stack_events,
                key=lambda e: e.self_cpu_time_total,
                reverse=True,
            )

        # Compute total time for percentage
        total_gpu_us = sum(
            e.self_device_time_total for e in self._prof.key_averages()
        )
        total_cpu_us = sum(
            e.self_cpu_time_total for e in self._prof.key_averages()
        )

        results = []
        for rank, evt in enumerate(stack_events_sorted[: self._top_n], 1):
            entry = {
                "rank": rank,
                "name": evt.key,
                "calls": evt.count,
                "self_gpu_time_ms": us_to_ms(evt.self_device_time_total),
                "self_cpu_time_ms": us_to_ms(evt.self_cpu_time_total),
                "gpu_time_total_ms": us_to_ms(evt.device_time_total),
                "cpu_time_total_ms": us_to_ms(evt.cpu_time_total),
            }

            # Percentage of total
            if total_gpu_us > 0:
                entry["pct_of_total_gpu"] = round(
                    (evt.self_device_time_total / total_gpu_us) * 100, 2
                )
            if total_cpu_us > 0:
                entry["pct_of_total_cpu"] = round(
                    (evt.self_cpu_time_total / total_cpu_us) * 100, 2
                )

            # Stack trace
            if hasattr(evt, 'stack') and evt.stack:
                entry["stack_trace"] = evt.stack

            # Input shapes
            if hasattr(evt, 'input_shapes') and evt.input_shapes:
                entry["input_shapes"] = [
                    list(s) if hasattr(s, '__iter__') else s
                    for s in evt.input_shapes
                ]

            # FLOPS and roofline
            if evt.flops > 0:
                entry["flops"] = evt.flops
                entry["gflops"] = round(evt.flops / 1e9, 2)

                roofline = compute_roofline(
                    flops=evt.flops,
                    gpu_time_us=evt.self_device_time_total,
                    gpu_name=self._gpu_name,
                    dtype=self._model_dtype,
                )
                if roofline:
                    entry["roofline"] = roofline

            # Memory
            self_cuda_mem = getattr(evt, 'self_cuda_memory_usage', 0) or 0
            if self_cuda_mem != 0:
                entry["self_cuda_memory_mb"] = bytes_to_mb(self_cuda_mem)
            self_cpu_mem = getattr(evt, 'self_cpu_memory_usage', 0) or 0
            if self_cpu_mem != 0:
                entry["self_cpu_memory_mb"] = bytes_to_mb(self_cpu_mem)

            results.append(entry)

        return results

    def _get_memory_allocators(self) -> Dict[str, Any]:
        """Find which operators allocate the most memory."""
        # Use stack-grouped for memory allocation traces
        try:
            events = self._prof.key_averages(group_by_stack_n=3)
        except Exception:
            events = self._prof.key_averages()

        # GPU memory allocators
        gpu_allocators = sorted(
            [e for e in events if (getattr(e, 'self_cuda_memory_usage', 0) or 0) > 0],
            key=lambda e: getattr(e, 'self_cuda_memory_usage', 0) or 0,
            reverse=True,
        )

        # CPU memory allocators
        cpu_allocators = sorted(
            [e for e in events if (getattr(e, 'self_cpu_memory_usage', 0) or 0) > 0],
            key=lambda e: getattr(e, 'self_cpu_memory_usage', 0) or 0,
            reverse=True,
        )

        def format_allocator(evt):
            entry = {
                "op": evt.key,
                "self_cuda_memory_mb": bytes_to_mb(getattr(evt, 'self_cuda_memory_usage', 0) or 0),
                "self_cpu_memory_mb": bytes_to_mb(getattr(evt, 'self_cpu_memory_usage', 0) or 0),
                "calls": evt.count,
            }
            if hasattr(evt, 'stack') and evt.stack:
                entry["stack_trace"] = evt.stack
            return entry

        return {
            "top_gpu_allocators": [
                format_allocator(e) for e in gpu_allocators[: 10]
            ],
            "top_cpu_allocators": [
                format_allocator(e) for e in cpu_allocators[: 10]
            ],
        }
