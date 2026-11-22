"""The control panel: metrics parsing, the planner, the supervisor (with a stand-in engine) and the HTTP API."""

import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

from seventy import memory
from seventy.config import load_config
from seventy.panel import hardware, planner, telemetry
from seventy.panel.supervisor import Settings, Supervisor

METRICS = """# HELP seventy:generation_tokens_total x
seventy:generation_tokens_total 1200
seventy:uptime_seconds 50
seventy:time_to_first_token_seconds_bucket{le="0.5"} 2
seventy:time_to_first_token_seconds_bucket{le="1.0"} 6
seventy:time_to_first_token_seconds_bucket{le="+Inf"} 10
seventy:time_to_first_token_seconds_count 10
"""


def test_parse_metrics_and_quantile():
    values, buckets = telemetry.parse_metrics(METRICS)
    assert values["seventy:generation_tokens_total"] == 1200
    b = buckets["seventy:time_to_first_token_seconds"]
    assert b[0] == (0.5, 2) and b[-1][0] == float("inf")
    assert telemetry.quantile(None, b, 0.5) == pytest.approx(0.5 + 0.5 * (3 / 4))
    prev = [(0.5, 2), (1.0, 6), (float("inf"), 10)]
    assert telemetry.quantile(prev, b, 0.5) is None


def test_settings_command_keeps_the_key_off_the_command_line(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere")
    s = Settings(model="/m", gpus=[2, 3], api_key="sk-secret", thinking=False, draft_model="/d")
    cmd = s.command()
    assert "sk-secret" not in " ".join(cmd)
    assert cmd[cmd.index("--gpus") + 1] == "2,3" and "--draft-model" in cmd
    assert json.loads(cmd[cmd.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    env = s.environment()
    assert env["SEVENTY_API_KEY"] == "sk-secret" and env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and "LD_LIBRARY_PATH" not in env
    assert s.public()["api_key"] == "set"


def fake_inventory(n=4, mem=32 << 30, cc=(7, 0)):
    gpus = [hardware.Gpu(i, "Tesla V100-SXM2-32GB", f"u{i}", f"00000000:{i:02x}:00.0", mem, cc,
                         {j: 2 for j in range(n) if j != i}) for i in range(n)]
    return hardware.Inventory("580", gpus, "cpu", 32, 128 << 30, 100 << 30, 1 << 40, "/")


@pytest.mark.model
def test_plan_for_the_real_checkpoint(model_path, tmp_path):
    inv = fake_inventory()
    s = Settings(model=str(model_path), gpus=[0, 1, 2, 3], max_model_len=262144)
    p = planner.plan(s, inv, planner.Profiles(tmp_path))
    assert p["errors"] == [] and p["ok"] and not p["measured"]
    assert 1_500_000 < p["kv_tokens"] < 3_000_000
    planner.Profiles(tmp_path).put(s, 32 << 30, {"gpu_total": 31 << 30, "weights": 5 << 30, "drafter": 0,
                                                 "graphs": 1 << 28, "overhead": 1 << 30})
    q = planner.plan(s, inv, planner.Profiles(tmp_path))
    assert q["measured"] and q["kv_tokens"] < p["kv_tokens"]
    bad = planner.plan(Settings(model=str(model_path), gpus=[0, 1, 2]), inv)
    assert any("1, 2, 4 or 8" in e for e in bad["errors"])
    old = planner.plan(Settings(model=str(model_path), gpus=[0]), fake_inventory(cc=(8, 0)))
    assert any("not a V100" in e for e in old["errors"])


def mixed_inventory(sizes):
    inv = fake_inventory(len(sizes))
    for g, size in zip(inv.gpus, sizes):
        g.memory_total = size
    return inv


@pytest.mark.model
def test_plan_for_gpus_of_different_sizes_uses_the_smallest(model_path):
    s = Settings(model=str(model_path), gpus=[0, 1, 2, 3], max_model_len=262144)
    mixed = planner.plan(s, mixed_inventory([32 << 30, 32 << 30, 16 << 30, 16 << 30]))
    small = planner.plan(s, fake_inventory(mem=16 << 30))
    assert mixed["errors"] == [] and mixed["ok"]
    assert mixed["gpu_bytes"] == small["gpu_bytes"] and mixed["kv_tokens"] == small["kv_tokens"]
    assert any("different memory sizes (16 GiB and 32 GiB)" in w for w in mixed["warnings"])
    assert not any("different memory sizes" in w for w in small["warnings"])


@pytest.mark.model
def test_a_profile_is_kept_per_gpu_memory_size(model_path, tmp_path):
    s = Settings(model=str(model_path), gpus=[0, 1, 2, 3], max_model_len=262144)
    profiles = planner.Profiles(tmp_path)
    profiles.put(s, 32 << 30, {"gpu_total": 31 << 30, "weights": 5 << 30, "drafter": 0, "graphs": 1 << 28,
                               "overhead": 1 << 30})
    assert planner.plan(s, fake_inventory(), profiles)["measured"]
    small = planner.plan(s, fake_inventory(mem=16 << 30), profiles)
    assert not small["measured"] and small["gpu_bytes"] < 16 << 30
    profiles.put(s, 16 << 30, {"gpu_total": 15 << 30, "weights": 5 << 30, "drafter": 0, "graphs": 1 << 28,
                               "overhead": 1 << 30})
    mixed = planner.plan(s, mixed_inventory([32 << 30, 16 << 30, 32 << 30, 16 << 30]), profiles)
    assert mixed["measured"] and mixed["gpu_bytes"] == 15 << 30


def test_prefix_cache_entries_are_counted_as_in_the_engine():
    assert memory.prefix_entries(8, True) == 24
    assert memory.prefix_entries(8, True, 5) == 5
    assert memory.prefix_entries(8, False, 5) == 0


@pytest.mark.model
def test_plan_counts_the_state_slots_the_engine_allocates(model_path):
    cfg = load_config(model_path)
    per_slot = memory.state_bytes_per_slot(cfg, 4)
    for prefix_cache, slots in ((True, 8 + 24), (False, 8)):
        s = Settings(model=str(model_path), gpus=[0, 1, 2, 3], max_num_seqs=8, prefix_cache=prefix_cache)
        assert planner.plan(s, fake_inventory())["state"] == (slots + 1) * per_slot


STUB = r'''
import http.server, sys, time
port = int(sys.argv[sys.argv.index("--port") + 1])
mode = open(sys.argv[-1]).read().strip() if len(sys.argv) > 1 else "serve"
print("seventy-stage: weights loading stub", flush=True)
time.sleep(0.3)
print("seventy-stage: graphs capturing", flush=True)
if mode == "die":
    time.sleep(1.0)
    sys.exit(3)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def stub_engine(tmp_path, monkeypatch):
    script = tmp_path / "stub.py"
    script.write_text(STUB)
    mode = tmp_path / "mode"
    mode.write_text("serve")
    monkeypatch.setattr(Settings, "command", lambda self: [sys.executable, str(script), "--port", str(self.port), str(mode)])
    return mode


def wait_for(sup, states, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if sup.status.state in states:
            return sup.status.state
        time.sleep(0.2)
    raise AssertionError(f"state {sup.status.state}, wanted {states}")


def test_supervisor_start_stop_crash_restart(tmp_path, stub_engine):
    sup = Supervisor(tmp_path / "home")
    s = Settings(model="/m", gpus=[0], port=free_port())
    sup.start(s)
    assert wait_for(sup, ("ready",)) == "ready"
    assert (tmp_path / "home" / "last_good.json").is_file()
    run_dir = tmp_path / "home" / "runs" / sup.status.run
    assert "seventy-stage: graphs" in (run_dir / "engine.log").read_text()
    with pytest.raises(RuntimeError):
        sup.start(s)
    sup.stop(timeout=10)
    assert sup.status.state == "cold" and sup.proc is None

    stub_engine.write_text("die")
    sup.auto_restart = True
    sup.start(s)
    wait_for(sup, ("crashed",))
    reports = sup.reports()
    assert reports and "exited with code 3" in reports[0]["reason"]
    assert "seventy-stage: weights" in sup.report(reports[0]["name"])["log_tail"]
    stub_engine.write_text("serve")
    assert wait_for(sup, ("ready",), timeout=40) == "ready"
    assert len(sup.status.restarts) == 1
    sup.stop(timeout=10)


def test_panel_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from seventy.panel import app as panel

    monkeypatch.setattr(hardware, "inventory", lambda *a, **k: fake_inventory())
    monkeypatch.setattr(hardware, "sample", lambda: [])
    open_app = panel.create_app("0.0.0.0", tmp_path / "net")
    with TestClient(open_app) as c:
        assert c.get("/api/state").status_code == 401
        token = (tmp_path / "net" / "ui_token").read_text().strip()
        r = c.get("/api/state", headers={"X-Seventy-Token": token})
        assert r.status_code == 200 and r.json()["token_required"] is True
        assert c.get("/").status_code == 200
    assert oct((tmp_path / "net").stat().st_mode & 0o777) == "0o700"
    local = panel.create_app("127.0.0.1", tmp_path / "local")
    with TestClient(local) as c:
        st = c.get("/api/state").json()
        assert st["status"]["state"] == "cold" and st["token_required"] is False
        assert "app.js" in c.get("/").text
        assert c.get("/static/app.css").status_code == 200
        assert c.get("/static/../app.py").status_code == 404
        r = c.post("/api/plan", json={"model": str(tmp_path / "nothing"), "gpus": [0]})
        assert r.status_code == 200 and r.json()["errors"]
        assert c.post("/api/start", json={"model": str(tmp_path / "nothing"), "gpus": [0]}).status_code == 400
        prefs = c.post("/api/prefs", json={"instance_name": "lab", "hf_endpoint": "https://example.invalid"}).json()
        assert prefs["instance_name"] == "lab"
        assert c.get("/api/state").json()["instance_name"] == "lab"
        assert c.post("/api/downloads", json={"repo": "not a repo"}).status_code == 400
        assert c.post("/api/api_key").json()["api_key"].startswith("sk-")
        assert isinstance(c.get("/api/history?seconds=300").json(), list)


TREE_STUB = r'''
import http.server, signal, subprocess, sys, time
port = int(sys.argv[sys.argv.index("--port") + 1])
worker = subprocess.Popen([sys.executable, "-c",
                           "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(600)"])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a):
        pass
try:
    http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
except KeyboardInterrupt:
    time.sleep(1.0)
    worker.terminate()
    worker.wait()
    print(f"stub: stopped cleanly, worker {worker.pid}", flush=True)
'''


def test_panel_shutdown_stops_the_engine_and_waits_for_its_processes(tmp_path, monkeypatch):
    """The panel's shutdown (Ctrl+C, systemctl stop) stops the engine and its workers before the panel exits."""
    from fastapi.testclient import TestClient

    from seventy.panel import app as panel

    script = tmp_path / "tree.py"
    script.write_text(TREE_STUB)
    monkeypatch.setattr(Settings, "command", lambda self: [sys.executable, str(script), "--port", str(self.port)])
    monkeypatch.setattr(hardware, "sample", lambda: [])
    app = panel.create_app("127.0.0.1", tmp_path / "home")
    with TestClient(app):
        sup = app.state.supervisor
        sup.start(Settings(model="/m", gpus=[0], port=free_port()))
        assert wait_for(sup, ("ready",)) == "ready"
        pid, run = sup.status.pid, sup.status.run
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    log = (tmp_path / "home" / "runs" / run / "engine.log").read_text()
    assert "stub: stopped cleanly" in log
    assert sup.status.state == "cold" and sup.status.exit_code == 0
