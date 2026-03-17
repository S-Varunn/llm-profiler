# `record_function` + Kineto + CUPTI Under the Hood

This document explains what happens when `record_function` is called, how PyTorch hands tracing off to Kineto and CUPTI, and how raw resource events become useful profiling metrics.

---

## 1) Big picture

When you run:

```python
with record_function("LLM_GENERATE"):
    with torch.no_grad():
        outputs = model.generate(...)
```

there are **two parallel signal paths**:

1. **CPU operator path (PyTorch callbacks)**
   - `RecordFunction` enter/exit callbacks collect CPU-side op spans.
2. **GPU activity path (Kineto + CUPTI)**
   - CUPTI collects GPU kernels, memcpy/memset, runtime/driver calls, sync, overhead.

Then both streams are correlated and exported as:
- aggregated tables (`key_averages()`), and
- Chrome trace (`export_chrome_trace`) for timeline-level analysis.

---

## 2) In your project: where tracing wraps generation

Your main wrapping point is here:
- [llm_inference_profiler/profiler.py](../llm_inference_profiler/profiler.py#L192-L207)

It runs:
- `torch.profiler.profile(...)`
- then `record_function("LLM_GENERATE")`
- then `model.generate(...)`

Afterward it:
- exports Chrome trace,
- runs `EventAnalyzer` and `TraceAnalyzer`,
- assembles `summary / diagnostics / drilldown` JSON.

Related local analyzers:
- [llm_inference_profiler/event_analyzer.py](../llm_inference_profiler/event_analyzer.py)
- [llm_inference_profiler/trace_analyzer.py](../llm_inference_profiler/trace_analyzer.py)
- [llm_inference_profiler/token_tracker.py](../llm_inference_profiler/token_tracker.py)
- [llm_inference_profiler/memory_tracker.py](../llm_inference_profiler/memory_tracker.py)

---

## 3) What happens when `record_function` is entered/exited

### Python layer

`record_function` context manager:
- enter calls `torch.ops.profiler._record_function_enter_new(...)`
- exit calls `torch.ops.profiler._record_function_exit._RecordFunction(...)`

Source:
- https://github.com/pytorch/pytorch/blob/v2.10.0/torch/autograd/profiler.py#L748-L821

### C++ bridge layer

`_record_function_enter_new` / `_record_function_exit` map to C++ ops:
- `record_function_enter_new(...)`
- `record_function_exit_new(...)`

Those call:
- `rec.before(...)` on enter
- `rec.end()` on exit

Source:
- https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/autograd/record_function_ops.cpp#L19-L75

### ATen `RecordFunction` internals

On enter:
- `RecordFunction::before(...)` stores name/scope/sequence and runs start callbacks.

On exit:
- `RecordFunction::end()` runs end callbacks and closes the span.

Source:
- https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/record_function.cpp#L727-L744
- https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/record_function.cpp#L553-L561

### Why this matters

Those callbacks are exactly where profiler hooks attach. In Kineto profiling mode, `onFunctionEnter` / `onFunctionExit` are registered and push events into `RecordQueue`.

Source:
- https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/autograd/profiler_kineto.cpp#L495-L522
- https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/autograd/profiler_kineto.cpp#L762-L804

---

## 4) How PyTorch transitions into Kineto tracing

### Python profiler start sequence

Profiler context does:
1. `_prepare_profiler(...)`
2. `_enable_profiler(...)`

Source:
- https://github.com/pytorch/pytorch/blob/v2.10.0/torch/autograd/profiler.py#L370-L383

### C++ bridge (`profiler_kineto.cpp`)

- `prepareProfiler(...)` calls `kineto::prepareTrace(...)`
- `enableProfiler(...)` pushes callbacks, then calls `kineto::startTrace()`

Source:
- https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/autograd/profiler_kineto.cpp#L620-L638
- https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/autograd/profiler_kineto.cpp#L796-L804

### Kineto shim (`kineto_shim.cpp`)

- `prepareTrace(...)` -> `libkineto::api().activityProfiler().prepareTrace(...)`
- `startTrace()` -> `libkineto::api().activityProfiler().startTrace()`
- `stopTrace()` -> `libkineto::api().activityProfiler().stopTrace()`

Source:
- https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/profiler/kineto_shim.cpp#L257-L342

---

## 5) What `libkineto::api()` actually is

`libkineto::api()` is a process-wide singleton (`LibkinetoApi`) that owns:
- active `ActivityProfilerInterface` implementation
- optional client bridge
- config loader

Source:
- https://github.com/pytorch/kineto/blob/main/libkineto/include/libkineto.h#L69-L105
- https://github.com/pytorch/kineto/blob/main/libkineto/include/libkineto.h#L155
- https://github.com/pytorch/kineto/blob/main/libkineto/src/libkineto_api.cpp#L16-L19

On init, Kineto registers `ActivityProfilerProxy` as the profiler implementation:
- https://github.com/pytorch/kineto/blob/main/libkineto/src/init.cpp#L171-L173

So `activityProfiler()` returns that proxy object, which forwards calls downstream.

---

## 6) What happens inside `startTrace()` in libkineto

The synchronous API contract is explicitly:

`prepareTrace -> startTrace -> stopTrace`

Source:
- https://github.com/pytorch/kineto/blob/main/libkineto/include/ActivityProfilerInterface.h#L44-L66

### Step-by-step call path

1. `ActivityProfilerProxy::startTrace()` forwards to controller
   - https://github.com/pytorch/kineto/blob/main/libkineto/src/ActivityProfilerProxy.cpp#L79-L84

2. `ActivityProfilerController::startTrace()` calls backend profiler
   - `profiler_->startTrace(now)`
   - https://github.com/pytorch/kineto/blob/main/libkineto/src/ActivityProfilerController.cpp#L408-L410

3. Backend chosen at construction:
   - `CuptiActivityProfiler` (NVIDIA/CUPTI)
   - `RocmActivityProfiler` (ROCm)
   - or `GenericActivityProfiler` (CPU-only)
   - https://github.com/pytorch/kineto/blob/main/libkineto/src/ActivityProfilerController.cpp#L82-L95

4. `GenericActivityProfiler::startTrace(...)` is lock-protected and calls `startTraceInternal(...)`
   - https://github.com/pytorch/kineto/blob/main/libkineto/src/GenericActivityProfiler.h#L159-L163
   - https://github.com/pytorch/kineto/blob/main/libkineto/src/GenericActivityProfiler.cpp#L604-L613

5. `startTraceInternal(...)` does:
   - mark capture start timestamp
   - start child profiler sessions (`session->start()`)
   - state: `Warmup -> CollectTrace`

Important nuance:
- CUPTI enablement usually happens during **configure/prepare** warmup, not only at start boundary.
- see `configure(...)->enableGpuTracing()` path:
  - https://github.com/pytorch/kineto/blob/main/libkineto/src/GenericActivityProfiler.cpp#L521-L535

---

## 7) Where resources are monitored (exact CUPTI calls)

### Activity stream enable/disable

In `CuptiActivityApi::enableCuptiActivities(...)`, Kineto calls:
- `cuptiActivityRegisterCallbacks(...)`
- `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY)`
- `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET)`
- `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)`
- `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION)`
- `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION)`
- `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME)`
- `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER)`
- `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD)`

Source:
- https://github.com/pytorch/kineto/blob/main/libkineto/src/CuptiActivityApi.cpp#L295-L363

Disable path uses matching `cuptiActivityDisable(...)` calls:
- https://github.com/pytorch/kineto/blob/main/libkineto/src/CuptiActivityApi.cpp#L372-L402

### Callback channel (runtime/driver callback API)

`CuptiCallbackApi` sets callback subscription and domains:
- `cuptiSubscribe(...)`
- `cuptiEnableCallback(...)`
- `cuptiEnableDomain(...)`

Source:
- https://github.com/pytorch/kineto/blob/main/libkineto/src/CuptiCallbackApi.cpp#L164-L313

### Flush / teardown behavior

Kineto forces flush before finalize:
- `cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)`

Source:
- https://github.com/pytorch/kineto/blob/main/libkineto/src/CuptiActivityApi.cpp#L439-L447

---

## 8) How raw events become meaningful metrics

### A) Correlation and event closure

- CPU-side op spans come from `RecordFunction` callbacks (`begin_op`/`end` timing).
- GPU-side activity records come asynchronously from CUPTI buffers.
- Correlation IDs (runtime launch ↔ kernel) are used to connect CPU launch and GPU execution.

### B) Stop and process

On stop, controller does:
1. `profiler_->stopTrace(...)`
2. `profiler_->processTrace(...)`
3. package into `ActivityTrace`

Source:
- https://github.com/pytorch/kineto/blob/main/libkineto/src/ActivityProfilerController.cpp#L446-L464

PyTorch side consumes stop trace during result assembly:
- `ActivityTraceWrapper(stopTrace())` in collection path
- https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/profiler/collection.cpp#L1205-L1221

### C) Aggregated and timeline outputs

- `key_averages()` produces per-op aggregate metrics.
- Chrome trace preserves per-event timeline with categories and args.
- Your `TraceAnalyzer` reads low-level fields from trace JSON (`kernel`, `cuda_runtime`, `gpu_memcpy`, etc.).

---

## 9) Mapping to your profiler JSON

In your implementation, `_assemble_json(...)` merges:
- token timings
- memory snapshots
- profiler event diagnostics
- trace-derived low-level kernel/correlation/memcpy details

Source:
- [llm_inference_profiler/profiler.py](../llm_inference_profiler/profiler.py#L286-L383)

### What each section mostly reflects

- `summary.latency`
  - Tokenizer + `TokenTimingProcessor` (`TTFT`, decode, ITL)
- `summary.memory`
  - CUDA allocator snapshots (`torch.cuda.memory_stats()`)
- `diagnostics.top_operators_*`
  - `prof.key_averages()` aggregation
- `diagnostics.top_cuda_kernels` and sync/memory ops
  - event-level analysis from key averages
- `diagnostics.trace_event_categories`
  - category counts from Chrome trace
- `drilldown.cuda_kernels`
  - kernel-level details (grid/block/registers/shared memory if present)
- `drilldown.cpu_gpu_gaps`
  - runtime launch to kernel queue-delay analysis
- `drilldown.gpu_memcpy`
  - memcpy/memset totals and implied bandwidth

---

## 10) Mental model you can use while reading traces

Use this rule:

1. `record_function` gives **semantic boundaries** (what high-level block/op).
2. callback queue gives **CPU-side operator spans** (who launched work, when).
3. CUPTI gives **GPU-side execution facts** (what actually ran and for how long).
4. correlation IDs connect launch to execution.
5. analyzers compute bottleneck views (top ops, queue gaps, memory transfer cost).

If CPU launch is cheap but queue delay is high, bottleneck is usually downstream GPU scheduling/contention.
If launch overhead is high with many tiny kernels, kernel fusion/graph capture often helps.
If memcpy dominates, data movement/layout is often the first optimization target.

---

## 11) Common caveats

- First trace can include one-time warmup overhead (allocator/init/callback setup).
- Enabling more activity kinds raises overhead and trace size.
- Some fields depend on driver/runtime versions and may be absent.
- Synchronization events are optional and may need explicit enablement.
- CPU and GPU clocks are correlated approximately; tiny boundaries can be noisy.

---

## 12) Suggested reading order (fast)

1. Section 3 (`record_function` enter/exit)
2. Section 4–6 (PyTorch -> Kineto `startTrace` chain)
3. Section 7 (exact CUPTI monitoring calls)
4. Section 8–9 (how this becomes your JSON)

That sequence gives a full loop from API call to final metrics.
