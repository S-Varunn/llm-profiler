"""
LLM Inference Profiler
======================
A passive profiling tool for analyzing LLM inference performance.
Outputs a consolidated JSON with three layers of detail:
  - Summary: latency, memory, throughput
  - Diagnostics: per-phase operator breakdown, per-layer timing
  - Drilldown: stack traces, roofline, anomaly analysis
"""

from .profiler import LLMProfiler

__all__ = ["LLMProfiler"]
__version__ = "0.1.0"
