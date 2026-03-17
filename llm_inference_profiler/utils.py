"""
Utility functions for the LLM Inference Profiler.
GPU info, roofline calculations, JSON serialization helpers.
"""

import torch
import datetime
import json
import math
import numpy as np
from typing import Any, Dict, List, Optional


def get_gpu_info() -> Dict[str, Any]:
    """Collect GPU hardware information."""
    if not torch.cuda.is_available():
        return {"available": False}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    return {
        "available": True,
        "device_index": device,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_mb": round(props.total_memory / (1024 ** 2), 1),
        "multi_processor_count": props.multi_processor_count,
        "clock_rate_mhz": round(props.clock_rate / 1000, 1) if hasattr(props, 'clock_rate') else None,
    }


def get_theoretical_peak_tflops(gpu_name: str, dtype: torch.dtype) -> Optional[float]:
    """
    Return approximate theoretical peak TFLOPS for known GPUs.
    These are rough estimates; actual peak depends on clock boost.
    """
    # Map of GPU name substring -> (fp32 TFLOPS, fp16 TFLOPS)
    gpu_peaks = {
        "A100": (19.5, 312.0),   # With tensor cores for fp16
        "A10G": (9.7, 31.2),
        "H100": (51.2, 989.0),   # SXM with tensor cores
        "H200": (51.2, 989.0),
        "L40": (90.5, 181.0),
        "V100": (7.8, 125.0),
        "T4": (8.1, 65.0),
        "3090": (35.6, 71.0),
        "4090": (82.6, 165.0),
        "4080": (48.7, 97.5),
        "3080": (29.8, 59.5),
    }

    for key, (fp32, fp16) in gpu_peaks.items():
        if key.lower() in gpu_name.lower():
            if dtype in (torch.float16, torch.bfloat16):
                return fp16
            return fp32

    return None


def compute_roofline(
    flops: int,
    gpu_time_us: float,
    gpu_name: str,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Compute roofline analysis for an operation.
    Returns achieved TFLOPS, theoretical peak, utilization, and diagnosis.
    """
    if flops <= 0 or gpu_time_us <= 0:
        return {}

    gpu_time_s = gpu_time_us / 1e6
    achieved_tflops = (flops / 1e12) / gpu_time_s
    peak_tflops = get_theoretical_peak_tflops(gpu_name, dtype)

    result = {
        "achieved_tflops": round(achieved_tflops, 3),
    }

    if peak_tflops is not None:
        utilization = (achieved_tflops / peak_tflops) * 100
        result["peak_tflops"] = peak_tflops
        result["compute_utilization_pct"] = round(utilization, 2)

        if utilization < 15:
            result["diagnosis"] = "memory_bandwidth_bound"
        elif utilization < 50:
            result["diagnosis"] = "partially_compute_bound"
        else:
            result["diagnosis"] = "compute_bound"

    return result


def compute_percentile_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, median, p90, p99, min, max from a list of values."""
    if not values:
        return {}

    arr = np.array(values)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
    }


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract model metadata."""
    num_params = sum(p.numel() for p in model.parameters())
    dtype = next(model.parameters()).dtype

    # Try to find number of transformer layers
    num_layers = None
    for name, module in model.named_modules():
        if hasattr(module, 'layers') and isinstance(module.layers, torch.nn.ModuleList):
            num_layers = len(module.layers)
            break

    # Get device
    device = str(next(model.parameters()).device)

    return {
        "num_parameters": num_params,
        "num_parameters_human": _human_readable_count(num_params),
        "dtype": str(dtype),
        "device": device,
        "num_layers": num_layers,
    }


def _human_readable_count(num: int) -> str:
    """Convert a large integer to human-readable format (e.g., 7B, 1.5B, 350M)."""
    if num >= 1e9:
        return f"{num / 1e9:.1f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    return str(num)


def safe_json_serialize(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return obj
    elif isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """Save a dict to a JSON file with proper serialization."""
    serializable = safe_json_serialize(data)
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=indent)


def us_to_ms(us: float) -> float:
    """Convert microseconds to milliseconds, rounded."""
    return round(us / 1000, 3)


def bytes_to_mb(b: int) -> float:
    """Convert bytes to megabytes, rounded."""
    return round(b / (1024 ** 2), 2)
