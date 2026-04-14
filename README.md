# LLM Inference Profiler

A comprehensive profiling toolkit for analyzing Large Language Model (LLM) inference performance on NVIDIA GPUs. Measures latency, throughput, memory usage, and GPU utilization with detailed kernel-level analysis powered by PyTorch Profiler and Kineto.

## Features

- **End-to-End Inference Profiling**: Captures complete LLM generation pipeline
- **Token-Level Timing**: Measures TTFT (Time To First Token) and ITL (Inter-Token Latency) 
- **GPU Memory Tracking**: Monitor memory allocation, peak usage, and fragmentation
- **Operator Analysis**: Breakdown of execution time by PyTorch operations
- **Kernel Analysis**: Detailed CUDA kernel execution metrics and grid/block configuration
- **CPU-GPU Correlation**: Trace synchronization points and queue delays
- **Layer-Level Statistics**: Per-layer latency and memory metrics via hooks
- **GPU Information**: Hardware specs (SM count, memory, bandwidth)
- **JSON Export**: Structured output with summary, diagnostics, and detailed analysis

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA Toolkit (for GPU profiling)
- Transformers 4.0+

### Setup

```bash
# Clone and navigate to repo
cd pytorch-profiler

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_inference_profiler import LLMProfiler

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Create profiler
profiler = LLMProfiler(model=model, tokenizer=tokenizer)

# Run inference and profile
output_text = profiler.generate(
    prompt="Explain neural networks in one sentence.",
    max_new_tokens=50,
    do_sample=False,
)

# Display results
profiler.summary()

# Save detailed results
profiler.save("profile_output.json")
```

### API Backends (vLLM, Ollama, OpenAI-compatible)

`LLMProfiler` now supports modular adapters.
For API-served models, use `OpenAICompatibleAdapter` with the same profiler
interface.

If your model is running via vLLM (`/v1/chat/completions`), use:

```bash
# 1) (Optional) install vLLM in your environment
pip install vllm

# 2) Start vLLM server (example)
vllm serve Qwen/Qwen3-4B --host 0.0.0.0 --port 8000

# 3) In another terminal, run API profiler example
export API_BASE_URL=http://127.0.0.1:8000
export API_PROVIDER=vllm
# Optional model override if multiple models are served
# export MODEL_NAME=Qwen/Qwen3-4B

python example_vllm_api_profile.py
```

For Ollama OpenAI-compatible mode:

```bash
export API_BASE_URL=http://127.0.0.1:11434/v1
export API_PROVIDER=ollama
# Optional model override
# export MODEL_NAME=llama3.1:8b

python example_vllm_api_profile.py
```

For deep under-the-hood profiling on vLLM (CUDA kernels, CUDA API traces),
run in hybrid mode and launch vLLM under Nsight Systems:

```bash
# Requires Nsight Systems (nsys) installed and on PATH
export PROFILE_MODE=hybrid
export API_BASE_URL=http://127.0.0.1:8000
export API_PROVIDER=vllm
export VLLM_SERVER_CMD="vllm serve Qwen/Qwen3-4B --host 127.0.0.1 --port 8000"
export DEEP_COLLECTOR=nsys

python example_vllm_api_profile.py
```

Hybrid mode output includes:
- regular profiler summary/diagnostics/drilldown from API timing
- `drilldown.external_profiler` with generated `nsys` artifact paths
- server log path for troubleshooting startup/profiling issues

### Same-Machine Wrapped vLLM (No Remote Adapter Flow)

If you want to run vLLM and profiling in the same machine/process path,
use the profiled wrapper server:

```bash
export MODEL_NAME=Qwen/Qwen3-4B
export HOST=0.0.0.0
export PORT=9000
python profiled_vllm_server.py
```

Then send requests to the wrapper endpoint:

```bash
curl -s http://127.0.0.1:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-4B",
        "messages": [{"role":"user","content":"Explain transformers in one sentence."}],
        "max_tokens": 64,
        "temperature": 0.0
    }'
```

For each request, the wrapper:
1. runs generation through local vLLM adapter
2. profiles execution with torch profiler + trace analyzers
3. saves per-request JSON under `profiled_requests/`
4. returns model response and profile path in the API response

You can fetch raw profile JSON back via:

```bash
curl -s http://127.0.0.1:9000/profiles/<request_id>
```

For full CUDA-level trace capture, launch the wrapper under Nsight Systems:

```bash
export MODEL_NAME=Qwen/Qwen3-4B
export HOST=0.0.0.0
export PORT=9000
python run_profiled_vllm_with_nsys.py
```

This will:
- start `profiled_vllm_server.py` under `nsys profile`
- emit `.nsys-rep` files in `deep_profiles/`
- annotate request phases with NVTX ranges
- embed collector metadata into each saved request profile JSON

### Use vLLM Built-In Torch Profiler and Still Produce This Repo's JSON

If you start vLLM with internal profiler enabled, you can still use this repo's
analysis flow by bridging the generated trace files:

```bash
vllm serve meta-llama/Llama-3-8B \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_traces"}'
```

Then run:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000
export VLLM_MODEL=meta-llama/Llama-3-8B
export VLLM_TRACE_DIR=./vllm_traces
python example_vllm_internal_torch_profile.py
```

The script will:
1. call `/start_profile`
2. run workload requests via `/v1/chat/completions`
3. call `/stop_profile`
4. parse the latest trace (`.json` or `.json.gz`)
5. save analyzer output to `vllm_internal_torch_profile_output.json`

This creates `vllm_profile_output.json` with:
- TTFT (time to first streamed token)
- ITL percentile stats from streaming chunks
- End-to-end latency and tokens/sec
- Optional `/metrics` snapshot from the vLLM server

Note: In-process HuggingFace profiling still provides deep operator/kernel/layer
analysis. API profiling provides request-level latency/throughput and backend
metrics from the server process.

## Usage

### Basic Profiling

```python
from llm_inference_profiler import LLMProfiler

profiler = LLMProfiler(model=model, tokenizer=tokenizer)

# Generate with profiling
output = profiler.generate(
    prompt="Your prompt here",
    max_new_tokens=50,
    do_sample=False,
)

# Get results
results = profiler.results()  # Dict with summary/diagnostics/drilldown
profiler.summary()             # Print human-readable summary
profiler.save("output.json")   # Export to JSON file
```

### Accessing Results Programmatically

```python
results = profiler.results()

# Top-level metrics
summary = results["summary"]
print(f"TTFT: {summary['ttft_ms']:.2f} ms")
print(f"ITL: {summary['itl_median_ms']:.2f} ms (median)")
print(f"Throughput: {summary['tokens_per_second']:.2f} tokens/sec")
print(f"Peak GPU Memory: {summary['peak_memory_gb']:.2f} GB")

# Operator-level breakdown
diagnostics = results["diagnostics"]
for op in diagnostics["top_operators"]:
    print(f"{op['name']}: {op['time_ms']:.2f} ms")

# Kernel-level details
drilldown = results["drilldown"]
for kernel in drilldown["cuda_kernels"][:5]:
    print(f"{kernel['name']}: {kernel['time_ms']:.2f} ms ({kernel['count']} times)")
```

## Output Structure

The profiler generates a structured JSON with three levels of detail:

### 1. Summary
High-level metrics for quick understanding:
- **ttft_ms** - Time to first token
- **itl_median_ms** - Median inter-token latency
- **itl_p95_ms** - 95th percentile ITL
- **tokens_per_second** - Generation throughput
- **peak_memory_gb** - Peak GPU memory usage
- **memory_reserved_gb** - Total reserved memory
- **gpu_model** - GPU device name
- **total_generated_tokens** - Number of tokens generated

### 2. Diagnostics
Operator and phase breakdown:
- **top_operators** - Slowest PyTorch operations
- **phase_breakdown** - Time per inference phase (attention, MLP, etc.)
- **kernel_summary** - Summary of CUDA kernel execution
- **memory_timeline** - Memory usage over time
- **synchronization_events** - CPU-GPU sync points

### 3. Drilldown
Detailed low-level analysis:
- **cuda_kernels** - All CUDA kernels with duration, grid/block config
- **cpu_gpu_gaps** - Queue delays between CPU launch and GPU execution
- **gpu_memcpy** - Memory transfer operations
- **layer_timings** - Per-layer execution metrics
- **token_timings** - Timestamp and latency for each generated token

## Project Structure

```
pytorch-profiler/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── example_profiler.py                 # Usage example
├── profile_output.json                 # Sample output
│
├── llm_inference_profiler/             # Main package
│   ├── __init__.py
│   ├── profiler.py                     # Main LLMProfiler class
│   ├── event_analyzer.py               # Analyzes PyTorch profiler events
│   ├── trace_analyzer.py               # Parses Chrome trace JSON
│   ├── token_tracker.py                # Tracks per-token timing
│   ├── memory_tracker.py               # GPU memory snapshots
│   ├── hooks.py                        # PyTorch hook registration
│   ├── utils.py                        # Utility functions
│   └── __pycache__/
│
└── docs/                               # Documentation
    └── record_function_kineto_cupti_under_the_hood.md
```

## Component Details

### LLMProfiler (`profiler.py`)
Main orchestrator class that:
- Wraps model.generate() with profiling instrumentation
- Coordinates all sub-components (hooks, trackers, analyzers)
- Aggregates results into structured JSON
- Handles model output capture without modification

### EventAnalyzer (`event_analyzer.py`)
Processes PyTorch profiler events to extract:
- Operator execution time breakdown
- Memory allocation patterns
- Compute phase identification
- Operator call counts and averages

### TraceAnalyzer (`trace_analyzer.py`)
Parses Chrome trace JSON (from PyTorch profiler) to identify:
- CUDA kernel execution times
- Kernel launch overhead
- CPU-GPU queue delays
- Memory copy operations
- GPU utilization gaps

### TokenTimingProcessor (`token_tracker.py`)
Captures per-token latency via:
- LogitsProcessor hook at decode time
- Timestamp recording at each new token
- TTFT vs ITL computation
- Percentile calculations

### MemoryTracker (`memory_tracker.py`)
Snapshots GPU memory at key points:
- Before/after inference
- Per-layer (via hooks)
- torch.cuda.memory_stats() details
- Peak memory, fragmentation

### LayerHooks (`hooks.py`)
Registers PyTorch hooks on model layers:
- CUDA event timing per layer
- Input/output shape tracking
- Forward pass duration

## Advanced Usage

### Custom Model Behavior

The profiler doesn't modify model outputs—it passively observes:

```python
profiler = LLMProfiler(model=model, tokenizer=tokenizer)

# Works with any generation config
output = profiler.generate(
    prompt="Your prompt",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    # ... any other generation parameters
)
```

### Analyzing Results

```python
import json

with open("profile_output.json") as f:
    results = json.load(f)

# Custom analysis
summary = results["summary"]
drilldown = results["drilldown"]

# Find slowest kernel
slowest_kernel = max(
    drilldown["cuda_kernels"],
    key=lambda k: k["duration_ms"]
)
print(f"Slowest kernel: {slowest_kernel['name']}")

# Analyze memory growth
memory_timeline = results["diagnostics"]["memory_timeline"]
for phase, memory in memory_timeline.items():
    print(f"{phase}: {memory:.2f} GB")
```

## Understanding the Profiling Pipeline

The profiler combines multiple profiling techniques:

1. **PyTorch Profiler** - Captures operator execution and CUDA events
2. **Kineto** (PyTorch's GPU profiling layer) - Provides CPU-GPU correlation
3. **CUPTI** (NVIDIA's profiling API) - Collects CUDA kernel metrics
4. **Custom Hooks** - Adds layer-level timing
5. **LogitsProcessor** - Captures per-token timing

For a deep understanding of the internals, see [docs/record_function_kineto_cupti_under_the_hood.md](docs/record_function_kineto_cupti_under_the_hood.md), which explains:
- How `record_function()` enters the profiler
- PyTorch → Kineto → libkineto → CUPTI call chain
- Exact CUDA monitoring API calls
- Mapping of low-level metrics to the final JSON output

## Performance Considerations

### Profiling Overhead
- Profiling adds 10-30% overhead depending on settings
- Kernel memory tracking adds minimal cost
- Full traces can be large (10-100MB) for long generation runs

### Optimizations
- Disable `with_stack=True` to reduce overhead if not needed
- Use `profile_memory=False` for speed if memory analysis not required
- Batch profile multiple runs and average results

## Troubleshooting

### CUDA Out of Memory
If profiling causes OOM:
1. Reduce `max_new_tokens`
2. Disable `profile_memory=True`
3. Use smaller model or quantized variant

### Missing GPU Data
If CUDA metrics are empty:
1. Verify CUDA is available: `torch.cuda.is_available()`
2. Check GPU supports Kineto (NVIDIA GPUs required)
3. Ensure CUDA toolkit matches PyTorch build

### Chrome Trace Parsing Issues
The trace analyzer expects PyTorch's Chrome trace format:
- Ensure `profile.export_chrome_trace()` was called
- Verify trace file is valid JSON

## Dependencies

Key packages:
- `torch>=2.3.0` - PyTorch with profiler and Kineto
- `transformers>=4.45.0` - HuggingFace model loading
- `numpy` - Numerical operations
- `tensorboard` - For trace visualization

See [requirements.txt](requirements.txt) for full list.

## References

- [PyTorch Profiler Documentation](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Kineto on GitHub](https://github.com/pytorch/kineto)
- [Chrome Tracing Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMfGLKSJRVnFQnIvW0Y/)
- [CUDA Profiling Tools Interface (CUPTI)](https://docs.nvidia.com/cuda/cupti/)

## License

Master thesis project

## Author

Varun
