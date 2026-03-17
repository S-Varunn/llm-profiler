"""
Memory tracker for LLM inference profiling.
Takes snapshots of GPU memory at key points and analyzes allocation patterns.
"""

import torch
from typing import Any, Dict, List, Optional
from .utils import bytes_to_mb


class MemoryTracker:
    """
    Tracks GPU memory usage by taking snapshots at key inference phases.
    Uses torch.cuda.memory_stats() — pure reads with zero overhead.
    """

    def __init__(self):
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._has_cuda = torch.cuda.is_available()

    def snapshot(self, label: str) -> None:
        """Take a memory snapshot with the given label."""
        if not self._has_cuda:
            return

        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()

        self._snapshots[label] = {
            "allocated_bytes": stats.get("allocated_bytes.all.current", 0),
            "allocated_peak_bytes": stats.get("allocated_bytes.all.peak", 0),
            "reserved_bytes": stats.get("reserved_bytes.all.current", 0),
            "reserved_peak_bytes": stats.get("reserved_bytes.all.peak", 0),
            "active_bytes": stats.get("active_bytes.all.current", 0),
            "active_peak_bytes": stats.get("active_bytes.all.peak", 0),
            "num_alloc_retries": stats.get("num_alloc_retries", 0),
            "num_ooms": stats.get("num_ooms", 0),
            "oversize_allocations_current": stats.get("oversize_allocations.current", 0),
        }

    def reset_peak(self) -> None:
        """Reset peak memory stats to track only the inference period."""
        if self._has_cuda:
            torch.cuda.reset_peak_memory_stats()

    def get_summary(self) -> Dict[str, Any]:
        """
        Build the memory summary section of the JSON output.
        Expects snapshots named: 'before_inference', 'after_prefill', 'after_inference'.
        """
        if not self._has_cuda:
            return {"available": False}

        before = self._snapshots.get("before_inference", {})
        after_prefill = self._snapshots.get("after_prefill", {})
        after = self._snapshots.get("after_inference", {})

        total_gpu_mem = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory

        before_alloc = before.get("allocated_bytes", 0)
        peak_alloc = after.get("allocated_peak_bytes", 0)
        after_prefill_alloc = after_prefill.get("allocated_bytes", 0)

        return {
            "before_inference_mb": bytes_to_mb(before_alloc),
            "peak_during_inference_mb": bytes_to_mb(peak_alloc),
            "inference_delta_mb": bytes_to_mb(peak_alloc - before_alloc),
            "after_prefill_mb": bytes_to_mb(after_prefill_alloc),
            "prefill_delta_mb": bytes_to_mb(after_prefill_alloc - before_alloc),
            "gpu_total_mb": bytes_to_mb(total_gpu_mem),
            "gpu_utilization_pct": round(
                (peak_alloc / total_gpu_mem) * 100, 2
            ) if total_gpu_mem > 0 else 0,
            "reserved_peak_mb": bytes_to_mb(
                after.get("reserved_peak_bytes", 0)
            ),
            "num_alloc_retries": after.get("num_alloc_retries", 0),
            "num_ooms": after.get("num_ooms", 0),
            "oversize_allocations": after.get("oversize_allocations_current", 0),
        }

    def get_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Return all raw snapshots for drill-down."""
        result = {}
        for label, snap in self._snapshots.items():
            result[label] = {
                k: bytes_to_mb(v) if "bytes" in k else v
                for k, v in snap.items()
            }
        return result
