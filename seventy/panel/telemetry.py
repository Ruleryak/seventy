"""Live readings and history for the dashboard. No PyTorch.

Every 2 seconds: the GPUs (NVML), the host (CPU, RAM) and, when the engine is ready, its /metrics. Counters
become rates against the previous reading; time-to-first-token quantiles come from the histogram's change over the
interval. One hour of 2-second samples stays in memory; 10-second averages go to SQLite (history.sqlite in the panel
home) and stay 30 days, so the history survives restarts of the panel and of the engine.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

from . import hardware

INTERVAL = 2.0
KEEP_S = 3600
STORE_EVERY = 5
RETENTION_S = 30 * 86400


def parse_metrics(text: str) -> tuple[dict[str, float], dict[str, list[tuple[float, float]]]]:
    """Prometheus text to (plain values by name, histogram buckets by name: [(le, cumulative count)])."""
    values: dict[str, float] = {}
    buckets: dict[str, list[tuple[float, float]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            v = float(value)
        except ValueError:
            continue
        if name.endswith("}") and "_bucket{" in name:
            base = name[: name.index("_bucket{")]
            le = name[name.index('le="') + 4 : name.rindex('"')]
            buckets.setdefault(base, []).append((float("inf") if le == "+Inf" else float(le), v))
        else:
            values[name.split("{")[0]] = v
    return values, buckets


def quantile(prev: list[tuple[float, float]] | None, cur: list[tuple[float, float]], q: float) -> float | None:
    """The q quantile of the observations between two cumulative histogram readings (linear within a bucket)."""
    before = dict(prev or [])
    delta = [(le, c - before.get(le, 0.0)) for le, c in cur]
    total = delta[-1][1] if delta else 0.0
    if total <= 0:
        return None
    target = q * total
    lower_le, lower_c = 0.0, 0.0
    for le, c in delta:
        if c >= target:
            if le == float("inf"):
                return lower_le
            span = c - lower_c
            return lower_le + (le - lower_le) * ((target - lower_c) / span if span > 0 else 1.0)
        lower_le, lower_c = le, c
    return lower_le


class Telemetry:
    def __init__(self, home: Path, engine_url):
        self.home = Path(home)
        self.engine_url = engine_url
        self.samples: deque[dict] = deque(maxlen=int(KEEP_S / INTERVAL))
        self.lock = threading.Lock()
        self.db_path = self.home / "history.sqlite"
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS samples (t REAL PRIMARY KEY, data TEXT NOT NULL)")
        self._prev_metrics: tuple[float, dict, dict] | None = None
        self._prev_cpu: tuple[int, int] | None = None
        self._pending: list[dict] = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="telemetry")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _cpu(self) -> float | None:
        try:
            parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            vals = [int(x) for x in parts]
        except (OSError, ValueError, IndexError):
            return None
        idle, total = vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)
        prev, self._prev_cpu = self._prev_cpu, (idle, total)
        if prev is None or total == prev[1]:
            return None
        return 100.0 * (1.0 - (idle - prev[0]) / (total - prev[1]))

    def _engine(self, now: float) -> dict | None:
        url = self.engine_url()
        if not url:
            self._prev_metrics = None
            return None
        try:
            with urllib.request.urlopen(url + "/metrics", timeout=2) as r:
                values, buckets = parse_metrics(r.read().decode())
        except Exception:
            return None
        prev, self._prev_metrics = self._prev_metrics, (now, values, buckets)
        g = lambda k: values.get(f"seventy:{k}", 0.0)
        out = {"running": g("num_requests_running"), "waiting": g("num_requests_waiting"),
               "kv_usage": g("kv_cache_usage"), "kv_tokens": g("kv_cache_tokens"), "step_ms": g("step_time_ms"),
               "requests": g("requests_total"), "preemptions": g("preemptions_total")}
        if prev is not None and now > prev[0] and g("uptime_seconds") >= prev[1].get("seventy:uptime_seconds", 0):
            dt = now - prev[0]
            d = lambda k: max(0.0, g(k) - prev[1].get(f"seventy:{k}", 0.0))
            out["gen_tps"] = d("generation_tokens_total") / dt
            out["prompt_tps"] = d("prompt_tokens_total") / dt
            out["finished_ps"] = d("requests_finished_total") / dt
            drafted, accepted = d("spec_draft_tokens_total"), d("spec_accepted_tokens_total")
            out["accept_rate"] = accepted / drafted if drafted else None
            rows = d("spec_verify_steps_total")
            out["tokens_per_step"] = (accepted + rows) / rows if rows else None
            hit, computed = d("prefix_cache_hit_tokens_total"), d("prompt_tokens_total")
            out["cache_hit"] = hit / (hit + computed) if hit + computed else None
            name = "seventy:time_to_first_token_seconds"
            if name in buckets:
                out["ttft_p50"] = quantile(prev[2].get(name), buckets[name], 0.5)
                out["ttft_p95"] = quantile(prev[2].get(name), buckets[name], 0.95)
        return out

    def read(self) -> dict:
        now = time.time()
        sample: dict = {"t": now, "cpu": self._cpu()}
        try:
            mem = hardware._meminfo()
            sample["ram_used"] = mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)
        except Exception:
            pass
        try:
            sample["gpus"] = [vars(g) for g in hardware.sample()]
        except Exception:
            sample["gpus"] = []
        sample["engine"] = self._engine(now)
        return sample

    def _loop(self) -> None:
        last_prune = 0.0
        while not self._stop.is_set():
            started = time.time()
            try:
                s = self.read()
                with self.lock:
                    self.samples.append(s)
                self._pending.append(s)
                if len(self._pending) >= STORE_EVERY:
                    self._store(self._pending)
                    self._pending = []
                if started - last_prune > 3600:
                    with self._db() as db:
                        db.execute("DELETE FROM samples WHERE t < ?", (started - RETENTION_S,))
                    last_prune = started
            except Exception:
                pass
            self._stop.wait(max(0.1, INTERVAL - (time.time() - started)))

    @staticmethod
    def _mean(samples: list[dict]) -> dict:
        def avg(values):
            vals = [v for v in values if isinstance(v, (int, float))]
            return sum(vals) / len(vals) if vals else None

        out = {"t": samples[-1]["t"], "cpu": avg(s.get("cpu") for s in samples),
               "ram_used": avg(s.get("ram_used") for s in samples)}
        n_gpu = max((len(s.get("gpus") or []) for s in samples), default=0)
        out["gpus"] = []
        for i in range(n_gpu):
            rows = [s["gpus"][i] for s in samples if len(s.get("gpus") or []) > i]
            out["gpus"].append({k: avg(r.get(k) for r in rows) for k in rows[0]} if rows else {})
        engines = [s["engine"] for s in samples if s.get("engine")]
        if engines:
            keys = set().union(*engines)
            out["engine"] = {k: (max(e.get(k) or 0 for e in engines) if k in ("ttft_p95",) else
                                 avg(e.get(k) for e in engines)) for k in keys}
        else:
            out["engine"] = None
        return out

    def _store(self, samples: list[dict]) -> None:
        row = self._mean(samples)
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO samples (t, data) VALUES (?, ?)", (row["t"], json.dumps(row)))

    def latest(self) -> dict | None:
        with self.lock:
            return self.samples[-1] if self.samples else None

    def history(self, seconds: float, points: int = 600) -> list[dict]:
        """Samples of the last `seconds`, thinned to about `points` (means of neighbours)."""
        since = time.time() - seconds
        if seconds <= KEEP_S:
            with self.lock:
                rows = [s for s in self.samples if s["t"] >= since]
        else:
            with self._db() as db:
                rows = [json.loads(d) for (d,) in db.execute("SELECT data FROM samples WHERE t >= ? ORDER BY t",
                                                             (since,))]
        if len(rows) <= points:
            return rows
        step = len(rows) / points
        return [self._mean(rows[int(i * step) : max(int(i * step) + 1, int((i + 1) * step))]) for i in range(points)]
