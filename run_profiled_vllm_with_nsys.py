"""
Launch the profiled vLLM wrapper under Nsight Systems.

This starts profiled_vllm_server.py as the target process, wrapped with
`nsys profile`, so each request can be correlated with a full CUDA timeline.

Usage:
  export MODEL_NAME=Qwen/Qwen3-4B
  export HOST=0.0.0.0
  export PORT=9000
  python run_profiled_vllm_with_nsys.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from llm_inference_profiler import NsysCollector, ServerLauncher


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9000"))
PROFILE_OUTPUT_DIR = os.getenv("PROFILE_OUTPUT_DIR", "profiled_requests")
NSYS_OUTPUT_DIR = os.getenv("NSYS_OUTPUT_DIR", "deep_profiles")
SERVER_LOG_FILE = os.getenv("SERVER_LOG_FILE", "profiled_vllm_server.log")
SERVER_COMMAND = os.getenv("SERVER_COMMAND", f"python profiled_vllm_server.py")
HEALTH_URL = os.getenv("HEALTH_URL", f"http://127.0.0.1:{PORT}/health")


def main() -> None:
    collector = NsysCollector(output_dir=NSYS_OUTPUT_DIR, output_base="profiled_vllm")
    if not collector.is_available():
        raise RuntimeError("nsys was not found on PATH. Install Nsight Systems first.")

    wrapped_command = collector.wrap_command(SERVER_COMMAND)
    wrapped_command_str = " ".join(wrapped_command)

    env = {
        "MODEL_NAME": MODEL_NAME,
        "HOST": HOST,
        "PORT": str(PORT),
        "PROFILE_OUTPUT_DIR": PROFILE_OUTPUT_DIR,
        "EXTERNAL_PROFILER_COLLECTOR": "nsys",
        "EXTERNAL_PROFILER_OUTPUT_PREFIX": str(Path(NSYS_OUTPUT_DIR) / "profiled_vllm"),
    }

    launcher = ServerLauncher(
        command=wrapped_command_str,
        log_file=SERVER_LOG_FILE,
        env=env,
    )

    print(f"Launching wrapper under Nsight Systems: {wrapped_command_str}")
    pid = launcher.start()
    print(f"Server pid: {pid}")

    try:
        ready = launcher.wait_until_ready(HEALTH_URL, timeout_sec=180)
        if not ready:
            raise RuntimeError(f"Server did not become ready at {HEALTH_URL}")

        print(f"Server ready at {HEALTH_URL}")
        print(f"Requests will save profiles under: {PROFILE_OUTPUT_DIR}")
        print(f"Nsight output prefix: {collector.output_prefix}")
        print("Press Ctrl-C to stop.")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        launcher.stop()
        artifacts = collector.finalize()
        print("\nNsight artifacts:")
        for file_path in artifacts.files:
            print(f"- {file_path}")
        for note in artifacts.notes:
            print(f"- {note}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
