"""
Token generation tracker using HuggingFace LogitsProcessor.
Measures Time to First Token (TTFT) and Inter-Token Latency (ITL)
without modifying any logit values — purely observational.
"""

import time
import torch
from typing import Any, Dict, List, Optional
from .utils import compute_percentile_stats


class TokenTimingProcessor:
    """
    A LogitsProcessor-compatible callable that records timestamps
    after each forward pass during model.generate().

    It does NOT modify logits — it only reads the wall clock.
    Returned logits are identical to input logits.
    """

    def __init__(self):
        self._timestamps: List[float] = []
        self._start_time: Optional[float] = None
        self._prefill_done: bool = False
        self._has_cuda = torch.cuda.is_available()

    def reset(self) -> None:
        """Reset state for a new generation."""
        self._timestamps = []
        self._start_time = None
        self._prefill_done = False

    def mark_start(self) -> None:
        """Record the start time (call this right before model.generate)."""
        if self._has_cuda:
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Called by HuggingFace generate() after each forward pass.
        Records a timestamp, returns scores UNMODIFIED.
        """
        if self._has_cuda:
            torch.cuda.synchronize()

        self._timestamps.append(time.perf_counter())
        return scores

    def get_results(self) -> Dict[str, Any]:
        """
        Compute timing results from recorded timestamps.

        Returns:
            Dict with TTFT, ITL stats, per-token timeline.
        """
        if not self._timestamps or self._start_time is None:
            return {}

        num_tokens = len(self._timestamps)

        # First timestamp = end of prefill + first token logit
        ttft_s = self._timestamps[0] - self._start_time
        ttft_ms = ttft_s * 1000

        # Inter-token latencies (decode phase)
        itl_values_ms = []
        for i in range(1, num_tokens):
            delta = (self._timestamps[i] - self._timestamps[i - 1]) * 1000
            itl_values_ms.append(delta)

        # Decode total time
        decode_time_ms = 0.0
        if num_tokens > 1:
            decode_time_ms = (self._timestamps[-1] - self._timestamps[0]) * 1000

        # Total generation time
        total_time_ms = (self._timestamps[-1] - self._start_time) * 1000

        # Tokens per second (decode only, excluding prefill)
        decode_tokens = num_tokens - 1
        tokens_per_second = 0.0
        if decode_time_ms > 0:
            tokens_per_second = decode_tokens / (decode_time_ms / 1000)

        # Build per-token timeline
        timeline = []
        # Token 0: prefill
        timeline.append({
            "token_index": 0,
            "phase": "prefill",
            "latency_ms": round(ttft_ms, 3),
        })
        # Tokens 1..N: decode
        for i, itl_ms in enumerate(itl_values_ms):
            entry = {
                "token_index": i + 1,
                "phase": "decode",
                "latency_ms": round(itl_ms, 3),
            }
            timeline.append(entry)

        # Mark anomalous tokens (> 2x median)
        if itl_values_ms:
            import numpy as np
            median_itl = float(np.median(itl_values_ms))
            for entry in timeline:
                if entry["phase"] == "decode" and entry["latency_ms"] > median_itl * 2:
                    entry["anomaly"] = True
                    entry["deviation_from_median"] = round(
                        entry["latency_ms"] / median_itl, 2
                    )

        return {
            "num_tokens_generated": num_tokens,
            "time_to_first_token_ms": round(ttft_ms, 3),
            "prefill_time_ms": round(ttft_ms, 3),
            "decode_time_ms": round(decode_time_ms, 3),
            "total_generation_time_ms": round(total_time_ms, 3),
            "tokens_per_second": round(tokens_per_second, 2),
            "inter_token_latency": compute_percentile_stats(itl_values_ms),
            "token_timeline": timeline,
        }

    @property
    def timestamps(self) -> List[float]:
        """Raw timestamps for external analysis."""
        return self._timestamps

    @property
    def start_time(self) -> Optional[float]:
        return self._start_time
