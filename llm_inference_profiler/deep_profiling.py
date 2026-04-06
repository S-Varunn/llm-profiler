"""
Utilities for collecting deep profiling artifacts when serving models via vLLM.

This module is intentionally backend-agnostic:
- Launches/controls a model server process
- Optionally wraps launch under an external profiler (e.g., Nsight Systems)
- Waits for server readiness
- Collects profiling artifact metadata for downstream JSON reports
"""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class DeepProfileArtifacts:
    enabled: bool
    collector: str
    output_dir: str
    files: List[str]
    notes: List[str]


class BaseDeepCollector:
    """Base class for external deep profiling collectors."""

    name = "none"

    def is_available(self) -> bool:
        return True

    def wrap_command(self, raw_command: str) -> List[str]:
        return shlex.split(raw_command)

    def finalize(self) -> DeepProfileArtifacts:
        return DeepProfileArtifacts(
            enabled=False,
            collector=self.name,
            output_dir="",
            files=[],
            notes=["Deep collector disabled."],
        )


class NsysCollector(BaseDeepCollector):
    """
    Nsight Systems wrapper.

    It launches the target command through `nsys profile` and attempts
    to generate summary stats when the run completes.
    """

    name = "nsys"

    def __init__(self, output_dir: str = "deep_profiles", output_base: str = "vllm_under_the_hood"):
        self.output_dir = output_dir
        self.output_base = output_base
        self.output_prefix = os.path.join(self.output_dir, self.output_base)
        self.notes: List[str] = []

    def is_available(self) -> bool:
        return shutil.which("nsys") is not None

    def wrap_command(self, raw_command: str) -> List[str]:
        os.makedirs(self.output_dir, exist_ok=True)
        return [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "-o",
            self.output_prefix,
            *shlex.split(raw_command),
        ]

    def finalize(self) -> DeepProfileArtifacts:
        files = sorted(glob.glob(f"{self.output_prefix}*"))

        rep_candidates = [f for f in files if f.endswith(".nsys-rep") or f.endswith(".qdrep")]
        if rep_candidates:
            rep_file = rep_candidates[0]
            if shutil.which("nsys") is not None:
                stats_prefix = f"{self.output_prefix}_stats"
                cmd = [
                    "nsys",
                    "stats",
                    "--report",
                    "cuda_gpu_kern_sum,cuda_api_sum,osrt_sum",
                    "--format",
                    "csv",
                    "--output",
                    stats_prefix,
                    rep_file,
                ]
                try:
                    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    files = sorted(glob.glob(f"{self.output_prefix}*")) + sorted(glob.glob(f"{stats_prefix}*"))
                    files = sorted(set(files))
                except Exception as exc:
                    self.notes.append(f"Failed to run 'nsys stats': {exc}")
        else:
            self.notes.append("No .nsys-rep/.qdrep file found after run.")

        return DeepProfileArtifacts(
            enabled=True,
            collector=self.name,
            output_dir=self.output_dir,
            files=files,
            notes=self.notes,
        )


class ServerLauncher:
    """Launch and control a model server command."""

    def __init__(
        self,
        command: str,
        log_file: str = "vllm_server.log",
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.command = command
        self.log_file = log_file
        self.env = env or {}
        self.cwd = cwd
        self._proc: Optional[subprocess.Popen] = None
        self._log_handle = None

    def start(self) -> int:
        merged_env = os.environ.copy()
        merged_env.update(self.env)

        self._log_handle = open(self.log_file, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            shlex.split(self.command),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            cwd=self.cwd,
            env=merged_env,
            preexec_fn=os.setsid,
        )
        return self._proc.pid

    def wait_until_ready(self, health_url: str, timeout_sec: int = 180) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                return False
            try:
                req = Request(health_url, method="GET")
                with urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except URLError:
                pass
            time.sleep(1)
        return False

    def stop(self, grace_sec: int = 15) -> None:
        if not self._proc:
            return

        if self._proc.poll() is not None:
            self._cleanup_log()
            return

        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            self._cleanup_log()
            return

        deadline = time.time() + grace_sec
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self._cleanup_log()
                return
            time.sleep(0.5)

        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._cleanup_log()

    def _cleanup_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.flush()
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
