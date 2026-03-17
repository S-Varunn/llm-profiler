"""
Chrome trace analyzer.
Parses the JSON trace exported by prof.export_chrome_trace() to extract:
  - CUDA kernel details (grid/block dims, registers, shared memory)
  - CPU-GPU correlation gaps (kernel launch overhead, queue delay)
  - Event category distribution
  - GPU memcpy/memset details
"""

import json
import os
from typing import Any, Dict, List, Optional
from .utils import us_to_ms


class TraceAnalyzer:
    """
    Parses a Chrome trace JSON file to extract low-level GPU information
    not directly accessible through the PyTorch profiler Python API.
    """

    def __init__(self, trace_path: str):
        self._trace_path = trace_path
        self._trace_data: Optional[Dict] = None
        self._events: List[Dict] = []

    def load(self) -> bool:
        """Load and parse the trace file. Returns True on success."""
        if not os.path.exists(self._trace_path):
            return False

        try:
            with open(self._trace_path, 'r') as f:
                self._trace_data = json.load(f)

            self._events = self._trace_data.get("traceEvents", [])
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def analyze(self) -> Dict[str, Any]:
        """Run full trace analysis."""
        if not self._events:
            if not self.load():
                return {"error": "Could not load trace file"}

        return {
            "event_categories": self._get_event_categories(),
            "cuda_kernels": self._get_cuda_kernel_details(),
            "cpu_gpu_gaps": self._get_cpu_gpu_gaps(),
            "gpu_memcpy": self._get_memcpy_details(),
        }

    def _get_event_categories(self) -> Dict[str, int]:
        """Count events by category — overview of trace contents."""
        categories = {}
        for event in self._events:
            cat = event.get("cat", "unknown")
            if cat:
                categories[cat] = categories.get(cat, 0) + 1

        # Sort by count descending
        return dict(
            sorted(categories.items(), key=lambda x: x[1], reverse=True)
        )

    def _get_cuda_kernel_details(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Extract detailed CUDA kernel information from trace events.
        Kernel events have cat='kernel' and contain grid/block dimensions.
        """
        kernel_events = [
            e for e in self._events
            if e.get("cat") == "kernel" and e.get("dur") is not None
        ]

        # Aggregate by kernel name
        kernel_agg: Dict[str, Dict[str, Any]] = {}
        for evt in kernel_events:
            name = evt.get("name", "unknown")
            dur = evt.get("dur", 0)
            args = evt.get("args", {})

            if name not in kernel_agg:
                kernel_agg[name] = {
                    "name": name,
                    "total_dur_us": 0,
                    "count": 0,
                    "min_dur_us": float('inf'),
                    "max_dur_us": 0,
                    # Capture details from first occurrence
                    "grid": args.get("grid", None),
                    "block": args.get("block", None),
                    "registers_per_thread": args.get("registers per thread", None),
                    "shared_memory_bytes": args.get("shared memory", None),
                    "stream": args.get("stream", None),
                }

            agg = kernel_agg[name]
            agg["total_dur_us"] += dur
            agg["count"] += 1
            agg["min_dur_us"] = min(agg["min_dur_us"], dur)
            agg["max_dur_us"] = max(agg["max_dur_us"], dur)

        # Sort by total duration
        sorted_kernels = sorted(
            kernel_agg.values(),
            key=lambda x: x["total_dur_us"],
            reverse=True,
        )

        results = []
        for k in sorted_kernels[:top_n]:
            entry = {
                "name": k["name"],
                "total_gpu_time_ms": us_to_ms(k["total_dur_us"]),
                "calls": k["count"],
                "avg_time_us": round(k["total_dur_us"] / k["count"], 2),
                "min_time_us": round(k["min_dur_us"], 2),
                "max_time_us": round(k["max_dur_us"], 2),
            }

            # Add kernel launch config if available
            if k["grid"] is not None:
                entry["grid"] = k["grid"]
            if k["block"] is not None:
                entry["block"] = k["block"]
            if k["registers_per_thread"] is not None:
                entry["registers_per_thread"] = k["registers_per_thread"]
            if k["shared_memory_bytes"] is not None:
                entry["shared_memory_bytes"] = k["shared_memory_bytes"]

            results.append(entry)

        return results

    def _get_cpu_gpu_gaps(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze CPU-to-GPU launch overhead by looking at correlation
        between 'cuda_runtime' events and 'kernel' events.

        The gap between a cudaLaunchKernel on CPU and the actual kernel
        start on GPU indicates queue delay.
        """
        # Find cudaLaunchKernel events
        launch_events = [
            e for e in self._events
            if e.get("cat") == "cuda_runtime"
            and "launch" in e.get("name", "").lower()
        ]

        # Find kernel events indexed by correlation id
        kernel_by_corr: Dict[int, Dict] = {}
        for e in self._events:
            if e.get("cat") == "kernel":
                corr_id = e.get("args", {}).get("correlation", None)
                if corr_id is not None:
                    kernel_by_corr[corr_id] = e

        gaps = []
        for launch in launch_events:
            corr_id = launch.get("args", {}).get("correlation", None)
            if corr_id is None:
                continue

            kernel = kernel_by_corr.get(corr_id)
            if kernel is None:
                continue

            launch_end = launch.get("ts", 0) + launch.get("dur", 0)
            kernel_start = kernel.get("ts", 0)

            queue_delay_us = kernel_start - launch_end

            gaps.append({
                "cpu_event": launch.get("name", "cudaLaunchKernel"),
                "gpu_kernel": kernel.get("name", "unknown"),
                "launch_overhead_us": round(launch.get("dur", 0), 2),
                "queue_delay_us": round(queue_delay_us, 2),
                "kernel_duration_us": round(kernel.get("dur", 0), 2),
            })

        # Sort by queue delay to find worst bottlenecks
        gaps.sort(key=lambda x: x["queue_delay_us"], reverse=True)

        # Compute aggregate stats
        if gaps:
            all_queue_delays = [g["queue_delay_us"] for g in gaps]
            all_launch_overheads = [g["launch_overhead_us"] for g in gaps]

            import numpy as np
            summary = {
                "total_launches": len(gaps),
                "avg_queue_delay_us": round(float(np.mean(all_queue_delays)), 2),
                "max_queue_delay_us": round(float(np.max(all_queue_delays)), 2),
                "avg_launch_overhead_us": round(
                    float(np.mean(all_launch_overheads)), 2
                ),
                "worst_gaps": gaps[:top_n],
            }
        else:
            summary = {"total_launches": 0, "worst_gaps": []}

        return summary

    def _get_memcpy_details(self) -> List[Dict[str, Any]]:
        """Extract GPU memory copy event details."""
        memcpy_events = [
            e for e in self._events
            if e.get("cat") in ("gpu_memcpy", "gpu_memset")
        ]

        # Aggregate by name
        agg: Dict[str, Dict[str, Any]] = {}
        for evt in memcpy_events:
            name = evt.get("name", "unknown")
            dur = evt.get("dur", 0)
            args = evt.get("args", {})

            if name not in agg:
                agg[name] = {
                    "name": name,
                    "total_dur_us": 0,
                    "count": 0,
                    "total_bytes": 0,
                }

            agg[name]["total_dur_us"] += dur
            agg[name]["count"] += 1
            agg[name]["total_bytes"] += args.get("bytes", 0)

        results = []
        for item in sorted(agg.values(), key=lambda x: x["total_dur_us"], reverse=True):
            entry = {
                "name": item["name"],
                "total_time_ms": us_to_ms(item["total_dur_us"]),
                "count": item["count"],
            }
            if item["total_bytes"] > 0:
                entry["total_mb"] = round(item["total_bytes"] / (1024**2), 2)
                # Bandwidth
                if item["total_dur_us"] > 0:
                    bw_gb_s = (item["total_bytes"] / 1e9) / (item["total_dur_us"] / 1e6)
                    entry["bandwidth_gb_s"] = round(bw_gb_s, 2)
            results.append(entry)

        return results
