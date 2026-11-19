"""Starts, watches and stops the engine process. No PyTorch.

States: cold (no engine), starting, ready (GET /health answers), stopping, crashed. The engine runs as
`python -m seventy engine ...` in its own process session, from its run folder (runs/<time>/ under the panel home,
which also holds its settings and log). Startup progress comes from the engine's "seventy-stage:" log lines.

Stop: SIGINT to the API process, which stops the engine process, which stops the tensor-parallel workers (they
ignore SIGINT, so all ranks tear down together); SIGKILL to the whole process session after 60 seconds.

A crash is an exit nobody asked for, or 8 failed health checks in a row once ready. The supervisor then writes a
report (settings, exit code, the end of the log, GPU readings) and, when automatic restart is on, starts again
with the last settings that reached ready: at most 3 times in 10 minutes.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = ("process", "workers", "weights", "drafter", "cache", "graphs", "server", "ready")


@dataclass
class Settings:
    model: str
    gpus: list[int]
    draft_model: str | None = None
    max_model_len: int | None = None
    max_num_seqs: int = 8
    gpu_memory_utilization: float = 0.90
    prefill_budget: int = 8192
    kv_cache_dtype: str = "int8"
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    served_model_name: str | None = None
    thinking: bool = True
    max_image_pixels: int | None = None
    max_images_per_request: int = 20
    max_videos_per_request: int = 1
    cuda_graphs: bool = True
    prefix_cache: bool = True

    @staticmethod
    def from_dict(raw: dict) -> "Settings":
        known = {f for f in Settings.__dataclass_fields__}
        s = Settings(**{k: v for k, v in raw.items() if k in known})
        s.gpus = [int(g) for g in s.gpus]
        return s

    def public(self) -> dict:
        d = asdict(self)
        d["api_key"] = "set" if self.api_key else None
        return d

    def command(self) -> list[str]:
        cmd = [sys.executable, "-m", "seventy", "engine", "--model", self.model,
               "--gpus", ",".join(map(str, self.gpus)), "--host", self.host, "--port", str(self.port),
               "--max-num-seqs", str(self.max_num_seqs), "--gpu-memory-utilization", str(self.gpu_memory_utilization),
               "--prefill-budget", str(self.prefill_budget), "--kv-cache-dtype", self.kv_cache_dtype,
               "--max-images-per-request", str(self.max_images_per_request),
               "--max-videos-per-request", str(self.max_videos_per_request)]
        if self.max_model_len:
            cmd += ["--max-model-len", str(self.max_model_len)]
        if self.draft_model:
            cmd += ["--draft-model", self.draft_model]
        if self.served_model_name:
            cmd += ["--served-model-name", self.served_model_name]
        if not self.thinking:
            cmd += ["--default-chat-template-kwargs", json.dumps({"enable_thinking": False})]
        if self.max_image_pixels:
            cmd += ["--max-image-pixels", str(self.max_image_pixels)]
        if not self.cuda_graphs:
            cmd.append("--no-cuda-graphs")
        if not self.prefix_cache:
            cmd.append("--no-prefix-cache")
        return cmd

    def environment(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "CUDA_VISIBLE_DEVICES")}
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("SEVENTY_API_KEY", None)
        if self.api_key:
            env["SEVENTY_API_KEY"] = self.api_key
        return env

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        return f"http://{host}:{self.port}"


@dataclass
class Status:
    state: str = "cold"
    stage: str | None = None
    detail: str = ""
    run: str | None = None
    started: float | None = None
    ready_at: float | None = None
    pid: int | None = None
    exit_code: int | None = None
    settings: dict | None = None
    restarts: list[float] = field(default_factory=list)
    last_report: str | None = None


class Supervisor:
    def __init__(self, home: Path, gpu_snapshot=None):
        self.home = Path(home)
        (self.home / "runs").mkdir(parents=True, exist_ok=True)
        (self.home / "reports").mkdir(parents=True, exist_ok=True)
        self.status = Status()
        self.settings: Settings | None = None
        self.proc: subprocess.Popen | None = None
        self.lock = threading.RLock()
        self.auto_restart = True
        self.gpu_snapshot = gpu_snapshot
        self._stop_requested = False
        self._watcher: threading.Thread | None = None

    @property
    def last_good_file(self) -> Path:
        return self.home / "last_good.json"

    def last_good(self) -> Settings | None:
        try:
            return Settings.from_dict(json.loads(self.last_good_file.read_text()))
        except (OSError, ValueError, TypeError):
            return None

    def log_path(self) -> Path | None:
        return self.home / "runs" / self.status.run / "engine.log" if self.status.run else None

    def start(self, settings: Settings) -> None:
        with self.lock:
            if self.status.state in ("starting", "ready", "stopping"):
                raise RuntimeError(f"the engine is {self.status.state}; stop it first")
            self._launch(settings)

    def _launch(self, settings: Settings) -> None:
        run = time.strftime("%Y%m%d-%H%M%S")
        folder = self.home / "runs" / run
        n = 1
        while folder.exists():
            folder = self.home / "runs" / f"{run}-{n}"
            n += 1
        folder.mkdir(parents=True)
        (folder / "settings.json").write_text(json.dumps(settings.public(), indent=2))
        log = open(folder / "engine.log", "ab", buffering=0)
        self.proc = subprocess.Popen(settings.command(), cwd=folder, env=settings.environment(), stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()
        self.settings = settings
        self._stop_requested = False
        restarts = self.status.restarts
        self.status = Status(state="starting", stage="process", detail="starting the engine process",
                             run=folder.name, started=time.time(), pid=self.proc.pid, settings=settings.public(),
                             restarts=restarts, last_report=self.status.last_report)
        self._watcher = threading.Thread(target=self._watch, args=(self.proc, folder), daemon=True)
        self._watcher.start()

    def stop(self, timeout: float = 60.0) -> None:
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                self.status.state = "cold"
                return
            self._stop_requested = True
            self.status.state, self.status.detail = "stopping", "stopping the engine"
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            pass
        self._kill_session(proc)
        proc.wait(10)
        with self.lock:
            self.status.state, self.status.exit_code, self.status.detail = "cold", proc.returncode, "stopped"
            self.proc = None

    @staticmethod
    def _kill_session(proc: subprocess.Popen) -> None:
        """SIGKILL every process left in the engine's session (workers keep GPU memory otherwise)."""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.5)

    def _health(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _watch(self, proc: subprocess.Popen, folder: Path) -> None:
        log = folder / "engine.log"
        offset = 0
        failures = 0
        settings = self.settings
        while True:
            code = proc.poll()
            offset = self._follow(log, offset)
            if code is not None:
                with self.lock:
                    if self.proc is not proc or self._stop_requested:
                        return
                    self.status.exit_code = code
                self._kill_session(proc)
                self._crash(folder, f"the engine exited with code {code}")
                return
            if self.status.state in ("starting", "ready"):
                if self._health(settings.url):
                    failures = 0
                    if self.status.state == "starting":
                        with self.lock:
                            self.status.state, self.status.stage = "ready", "ready"
                            self.status.ready_at, self.status.detail = time.time(), "serving"
                        self.last_good_file.write_text(json.dumps(asdict(settings)))
                elif self.status.state == "ready":
                    failures += 1
                    if failures >= 8:
                        self._kill_session(proc)
                        proc.wait(30)
                        with self.lock:
                            self.status.exit_code = proc.returncode
                        self._crash(folder, "the engine stopped answering /health")
                        return
            time.sleep(2.0)

    def _follow(self, log: Path, offset: int) -> int:
        try:
            with open(log, "rb") as f:
                f.seek(offset)
                data = f.read()
        except OSError:
            return offset
        if not data:
            return offset
        text = data.decode(errors="replace")
        end = text.rfind("\n")
        if end < 0:
            return offset
        for line in text[: end + 1].splitlines():
            if "seventy-stage:" in line:
                stage, _, detail = line.split("seventy-stage:", 1)[1].strip().partition(" ")
                if stage in STAGES:
                    with self.lock:
                        if self.status.state == "starting":
                            self.status.stage, self.status.detail = stage, detail.strip()
            elif "Uvicorn running on" in line:
                with self.lock:
                    if self.status.state == "starting":
                        self.status.stage = "server"
        return offset + len(text[: end + 1].encode())

    def _crash(self, folder: Path, reason: str) -> None:
        tail = ""
        try:
            tail = "\n".join((folder / "engine.log").read_text(errors="replace").splitlines()[-200:])
        except OSError:
            pass
        report = {"time": time.time(), "run": folder.name, "reason": reason, "exit_code": self.status.exit_code,
                  "stage": self.status.stage, "settings": self.status.settings, "log_tail": tail}
        if self.gpu_snapshot is not None:
            try:
                report["gpus"] = self.gpu_snapshot()
            except Exception as e:
                report["gpus"] = f"unavailable: {e}"
        name = f"{folder.name}.json"
        (self.home / "reports" / name).write_text(json.dumps(report, indent=2))
        with self.lock:
            self.status.state, self.status.detail, self.status.last_report = "crashed", reason, name
            self.proc = None
            now = time.time()
            self.status.restarts = [t for t in self.status.restarts if now - t < 600]
            good = self.last_good()
            again = self.auto_restart and good is not None and len(self.status.restarts) < 3
            if again:
                self.status.restarts.append(now)
        if again:
            time.sleep(5.0)
            with self.lock:
                if self.status.state == "crashed":
                    self._launch(good)

    def reports(self) -> list[dict]:
        out = []
        for f in sorted((self.home / "reports").glob("*.json"), reverse=True)[:50]:
            try:
                r = json.loads(f.read_text())
                out.append({"name": f.name, "time": r.get("time"), "reason": r.get("reason"), "stage": r.get("stage")})
            except (OSError, ValueError):
                continue
        return out

    def report(self, name: str) -> dict | None:
        f = self.home / "reports" / Path(name).name
        try:
            return json.loads(f.read_text())
        except (OSError, ValueError):
            return None

    def log_tail(self, lines: int = 400) -> str:
        p = self.log_path()
        if p is None or not p.is_file():
            return ""
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 512 * 1024))
            text = f.read().decode(errors="replace")
        return "\n".join(text.splitlines()[-lines:])
