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
from .adapters import BaseInferenceAdapter, OpenAICompatibleAdapter, LocalVLLMAdapter
from .deep_profiling import BaseDeepCollector, NsysCollector, ServerLauncher

__all__ = [
  "LLMProfiler",
  "BaseInferenceAdapter",
  "OpenAICompatibleAdapter",
  "LocalVLLMAdapter",
  "BaseDeepCollector",
  "NsysCollector",
  "ServerLauncher",
]
__version__ = "0.1.0"
