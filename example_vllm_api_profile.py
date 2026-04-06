"""
Profile OpenAI-compatible API backends (vLLM, Ollama, and others)
through the unified LLMProfiler interface.

Examples:
  # vLLM
  export API_BASE_URL=http://127.0.0.1:8000
  export API_PROVIDER=vllm
  python example_vllm_api_profile.py

  # Ollama (OpenAI-compatible endpoint)
  export API_BASE_URL=http://127.0.0.1:11434/v1
  export API_PROVIDER=ollama
  python example_vllm_api_profile.py

    # Hybrid under-the-hood profiling (launch vLLM under Nsight Systems)
    export PROFILE_MODE=hybrid
    export VLLM_SERVER_CMD="vllm serve Qwen/Qwen3-4B --host 127.0.0.1 --port 8000"
    export DEEP_COLLECTOR=nsys
    python example_vllm_api_profile.py
"""

import os
import sys
from typing import Any, Dict
from urllib.error import HTTPError, URLError

from llm_inference_profiler import (
        LLMProfiler,
        OpenAICompatibleAdapter,
        BaseDeepCollector,
        NsysCollector,
        ServerLauncher,
)


API_BASE_URL = os.getenv("API_BASE_URL", os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"))
API_PROVIDER = os.getenv("API_PROVIDER", "vllm")
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("VLLM_MODEL", ""))
PROMPT = os.getenv("PROMPT", "Explain what a neural network is in one sentence.")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "vllm_profile_output.json")
PROFILE_MODE = os.getenv("PROFILE_MODE", "api").lower()  # api | hybrid
VLLM_SERVER_CMD = os.getenv("VLLM_SERVER_CMD", "")
SERVER_LOG_FILE = os.getenv("SERVER_LOG_FILE", "vllm_server.log")
DEEP_COLLECTOR = os.getenv("DEEP_COLLECTOR", "none").lower()  # none | nsys
DEEP_OUTPUT_DIR = os.getenv("DEEP_OUTPUT_DIR", "deep_profiles")


def _build_deep_collector() -> BaseDeepCollector:
    if PROFILE_MODE != "hybrid":
        return BaseDeepCollector()

    if DEEP_COLLECTOR == "nsys":
        return NsysCollector(output_dir=DEEP_OUTPUT_DIR)

    return BaseDeepCollector()


def main() -> None:
    launcher = None
    collector = _build_deep_collector()

    if PROFILE_MODE == "hybrid":
        if not VLLM_SERVER_CMD:
            raise RuntimeError(
                "PROFILE_MODE=hybrid requires VLLM_SERVER_CMD to launch the server under profiler control."
            )

        if not collector.is_available():
            raise RuntimeError(
                f"Deep collector '{collector.name}' is not available on this machine."
            )

        wrapped = collector.wrap_command(VLLM_SERVER_CMD)
        launcher = ServerLauncher(
            command=" ".join(wrapped),
            log_file=SERVER_LOG_FILE,
        )
        pid = launcher.start()
        print(f"Started server process (pid={pid})")

        ready = launcher.wait_until_ready(f"{API_BASE_URL}/health", timeout_sec=180)
        if not ready:
            launcher.stop()
            raise RuntimeError("Server did not become ready in time. See server log for details.")

    adapter = OpenAICompatibleAdapter(
        base_url=API_BASE_URL,
        model_name=MODEL_NAME,
        provider=API_PROVIDER,
        timeout_sec=300,
        collect_metrics=True,
    )

    profiler = LLMProfiler(adapter=adapter)

    try:
        output_text = profiler.generate(
            prompt=PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=TEMPERATURE > 0.0,
            temperature=TEMPERATURE,
        )
    finally:
        if launcher is not None:
            launcher.stop()

    profiler.summary()

    results: Dict[str, Any] = profiler.results() or {}
    if PROFILE_MODE == "hybrid":
        artifacts = collector.finalize()
        extra = {
            "drilldown": {
                "external_profiler": {
                    "enabled": artifacts.enabled,
                    "collector": artifacts.collector,
                    "output_dir": artifacts.output_dir,
                    "files": artifacts.files,
                    "notes": artifacts.notes,
                    "server_log": SERVER_LOG_FILE,
                }
            }
        }
        profiler.merge_results(extra)
        results = profiler.results() or {}

    profiler.save(OUTPUT_FILE)

    print("\nGenerated text:\n")
    print(output_text)
    print(f"\nSaved: {OUTPUT_FILE}")
    if PROFILE_MODE == "hybrid":
        files = results.get("drilldown", {}).get("external_profiler", {}).get("files", [])
        print("\nDeep profiling artifacts:")
        for f in files:
            print(f"- {f}")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.reason}")
        raise
    except URLError as exc:
        print(f"Connection error: {exc.reason}")
        raise
    except RuntimeError as exc:
        print(f"Runtime error: {exc}")
        sys.exit(1)
