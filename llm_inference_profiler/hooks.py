"""
Module hooks for layer-level and sub-layer timing.
Uses register_forward_pre_hook / register_forward_hook with CUDA events
to measure time spent in each transformer layer and its sub-components
(attention, MLP, normalization).
"""

import torch
from typing import Any, Dict, List, Optional, Tuple
from .utils import us_to_ms


class LayerHooks:
    """
    Attaches CUDA event-based timing hooks to transformer layers.

    Hooks record start/end CUDA events on each forward pass.
    After generation, call `compute_results()` to synchronize events
    and compute elapsed times.

    This is passive: hooks do not modify any tensors.
    """

    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._handles: List[Any] = []
        self._events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._has_cuda = torch.cuda.is_available()
        self._layer_names: List[str] = []

        # Discover model architecture and attach hooks
        self._attach_hooks()

    def _attach_hooks(self) -> None:
        """
        Walk the model tree and attach hooks to:
        - Embedding layer
        - Each transformer layer (and sub-components: attention, MLP, norms)
        - LM head
        """
        model = self._model

        # Find the actual model inside wrapper (e.g., model.model for HF models)
        inner_model = None
        if hasattr(model, 'model'):
            inner_model = model.model
        elif hasattr(model, 'transformer'):
            inner_model = model.transformer

        # --- Embedding ---
        embed_module = self._find_embedding(model, inner_model)
        if embed_module is not None:
            self._hook_module("embedding", embed_module)

        # --- Transformer Layers ---
        layers = self._find_transformer_layers(model, inner_model)
        if layers is not None:
            for i, layer in enumerate(layers):
                layer_key = f"layer_{i}"
                self._layer_names.append(layer_key)
                self._hook_module(layer_key, layer)

                # Sub-components within each layer
                self._hook_sublayer(layer, i, "self_attn", "attention")
                self._hook_sublayer(layer, i, "mlp", "mlp")

                # Norms — try common naming conventions
                for norm_attr in ("input_layernorm", "post_attention_layernorm",
                                  "ln_1", "ln_2", "layer_norm1", "layer_norm2",
                                  "norm1", "norm2"):
                    self._hook_sublayer(layer, i, norm_attr, "norm")

        # --- LM Head ---
        lm_head = self._find_lm_head(model)
        if lm_head is not None:
            self._hook_module("lm_head", lm_head)

    def _find_embedding(
        self, model: torch.nn.Module, inner_model: Optional[torch.nn.Module]
    ) -> Optional[torch.nn.Module]:
        """Locate the embedding layer."""
        for candidate in [inner_model, model]:
            if candidate is None:
                continue
            for attr in ("embed_tokens", "wte", "word_embeddings",
                         "embed_in", "embeddings"):
                if hasattr(candidate, attr):
                    return getattr(candidate, attr)
        return None

    def _find_transformer_layers(
        self, model: torch.nn.Module, inner_model: Optional[torch.nn.Module]
    ) -> Optional[torch.nn.ModuleList]:
        """Locate the transformer layer list."""
        for candidate in [inner_model, model]:
            if candidate is None:
                continue
            for attr in ("layers", "h", "block", "blocks", "decoder_layers"):
                if hasattr(candidate, attr):
                    layers = getattr(candidate, attr)
                    if isinstance(layers, torch.nn.ModuleList):
                        return layers
        return None

    def _find_lm_head(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Locate the language model head."""
        for attr in ("lm_head", "output", "output_projection"):
            if hasattr(model, attr):
                return getattr(model, attr)
        return None

    def _hook_sublayer(
        self,
        layer: torch.nn.Module,
        layer_idx: int,
        attr_name: str,
        component_type: str,
    ) -> None:
        """Hook a sub-component of a transformer layer if it exists."""
        if hasattr(layer, attr_name):
            sub_module = getattr(layer, attr_name)
            key = f"layer_{layer_idx}_{component_type}_{attr_name}"
            self._hook_module(key, sub_module)

    def _hook_module(self, key: str, module: torch.nn.Module) -> None:
        """Attach pre/post forward hooks with CUDA event recording."""
        self._events[key] = []

        if self._has_cuda:
            def pre_hook(mod, inp, _key=key):
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                # Store start event temporarily on the module
                mod._profile_start_event = start

            def post_hook(mod, inp, out, _key=key):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                start = getattr(mod, '_profile_start_event', None)
                if start is not None:
                    self._events[_key].append((start, end))
                    del mod._profile_start_event

            h1 = module.register_forward_pre_hook(pre_hook)
            h2 = module.register_forward_hook(post_hook)
            self._handles.extend([h1, h2])
        else:
            # CPU fallback using perf_counter
            import time

            def pre_hook(mod, inp, _key=key):
                mod._profile_start_time = time.perf_counter()

            def post_hook(mod, inp, out, _key=key):
                start = getattr(mod, '_profile_start_time', None)
                if start is not None:
                    elapsed_us = (time.perf_counter() - start) * 1e6
                    self._events[_key].append(elapsed_us)
                    del mod._profile_start_time

            h1 = module.register_forward_pre_hook(pre_hook)
            h2 = module.register_forward_hook(post_hook)
            self._handles.extend([h1, h2])

    def compute_results(self) -> Dict[str, Any]:
        """
        Synchronize CUDA events and compute layer-level timing.

        Returns a dict with:
        - summary: pct breakdown by component type (attention, mlp, norm, etc.)
        - per_layer: list of per-layer timing dicts
        """
        if self._has_cuda:
            torch.cuda.synchronize()

        timings: Dict[str, List[float]] = {}

        for key, event_list in self._events.items():
            elapsed_list = []
            for pair in event_list:
                if self._has_cuda:
                    start_evt, end_evt = pair
                    try:
                        elapsed_ms = start_evt.elapsed_time(end_evt)
                        elapsed_list.append(elapsed_ms)
                    except RuntimeError:
                        pass
                else:
                    # CPU: pair is already elapsed_us
                    elapsed_list.append(pair / 1000.0)

            timings[key] = elapsed_list

        return self._build_layer_breakdown(timings)

    def _build_layer_breakdown(
        self, timings: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Structure timing data into the layer_breakdown JSON section."""

        # Aggregate by component type
        total_attention_ms = 0.0
        total_mlp_ms = 0.0
        total_norm_ms = 0.0
        total_embedding_ms = 0.0
        total_lm_head_ms = 0.0

        # Per-layer data
        per_layer = []
        num_layers = len(self._layer_names)

        for layer_key in self._layer_names:
            layer_idx = int(layer_key.split("_")[1])
            layer_total = sum(timings.get(layer_key, []))

            # Find sub-component timings for this layer
            attn_ms = 0.0
            mlp_ms = 0.0
            norm_ms = 0.0

            for key, vals in timings.items():
                if not key.startswith(f"layer_{layer_idx}_"):
                    continue
                total = sum(vals)
                if "_attention_" in key:
                    attn_ms += total
                elif "_mlp_" in key:
                    mlp_ms += total
                elif "_norm_" in key:
                    norm_ms += total

            total_attention_ms += attn_ms
            total_mlp_ms += mlp_ms
            total_norm_ms += norm_ms

            per_layer.append({
                "layer": layer_idx,
                "total_ms": round(layer_total, 3),
                "attention_ms": round(attn_ms, 3),
                "mlp_ms": round(mlp_ms, 3),
                "norm_ms": round(norm_ms, 3),
                "forward_passes": len(timings.get(layer_key, [])),
            })

        # Embedding and LM head
        total_embedding_ms = sum(timings.get("embedding", []))
        total_lm_head_ms = sum(timings.get("lm_head", []))

        # Compute percentages
        grand_total = (
            total_attention_ms + total_mlp_ms + total_norm_ms
            + total_embedding_ms + total_lm_head_ms
        )

        def pct(val):
            return round((val / grand_total) * 100, 2) if grand_total > 0 else 0.0

        summary = {
            "attention_pct": pct(total_attention_ms),
            "mlp_pct": pct(total_mlp_ms),
            "normalization_pct": pct(total_norm_ms),
            "embedding_pct": pct(total_embedding_ms),
            "lm_head_pct": pct(total_lm_head_ms),
            "attention_total_ms": round(total_attention_ms, 3),
            "mlp_total_ms": round(total_mlp_ms, 3),
            "normalization_total_ms": round(total_norm_ms, 3),
            "embedding_total_ms": round(total_embedding_ms, 3),
            "lm_head_total_ms": round(total_lm_head_ms, 3),
        }

        return {
            "summary": summary,
            "per_layer": per_layer,
        }

    def remove_hooks(self) -> None:
        """Remove all registered hooks from the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        # Clean up any leftover temp attributes
        for module in self._model.modules():
            if hasattr(module, '_profile_start_event'):
                del module._profile_start_event
            if hasattr(module, '_profile_start_time'):
                del module._profile_start_time

    @property
    def layer_count(self) -> int:
        return len(self._layer_names)
