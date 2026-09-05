import os
import subprocess
import sys
import time
import atexit
from typing import Optional

import requests


class VLLMServer:
    def __init__(
        self,
        model: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.verbose = verbose
        self.base_url = f"http://{host}:{port}/v1"
        self._proc: Optional[subprocess.Popen] = None

        self.start(wait=True)

    def _build_cmd(self) -> list[str]:
        while True:
            try:
                if requests.get(f"{self.base_url}/models", timeout=2).ok:
                    print(
                        f"vLLM is already running on {self.base_url}, will start another instance on port {self.port + 1}"
                    )
                    self.port += 1
                    self.base_url = f"http://{self.host}:{self.port}/v1"
                else:
                    break
            except Exception:
                break

        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            "1",
        ]

        if self.model.startswith("Qwen/"):
            cmd += [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
                "--reasoning-parser",
                "qwen3",
                "--enforce-eager",
            ]
        return cmd

    def _wait_for_ready(self, timeout: int = 1800) -> None:
        print(f"Waiting for {self.base_url} ...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                if requests.get(f"{self.base_url}/models", timeout=2).ok:
                    print("✅ vLLM is ready!")
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError("vLLM did not start in time")

    def start(self, wait: bool = True) -> None:
        if self._proc and self._proc.poll() is None:
            if wait:
                self._wait_for_ready()
            return

        print("Launching vLLM:", " ".join(self._build_cmd()))
        self._proc = subprocess.Popen(
            self._build_cmd(),
            stdout=None if self.verbose else subprocess.DEVNULL,
            stderr=None if self.verbose else subprocess.DEVNULL,
        )
        atexit.register(self.stop)

        if wait:
            self._wait_for_ready()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            print("\nStopping vLLM...")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
