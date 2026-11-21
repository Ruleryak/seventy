"""The control panel's HTTP interface and web UI. No PyTorch.

The panel home (default $XDG_STATE_HOME/seventy, else ~/.local/state/seventy) holds the settings, run folders,
error reports, measured profiles and the dashboard history. When the panel listens on an address other than
loopback, every /api/ request needs the UI token (header X-Seventy-Token or ?token=), which the panel prints at
start and keeps in ui_token in the home.

On SIGINT or SIGTERM (Ctrl+C, systemctl stop) the panel answers the open requests for at most 10 seconds, then
stops the engine and waits for all its processes (seventy.panel.supervisor), then exits. Thus a service manager
that signals only the panel process (systemd KillMode=mixed) gets a clean stop of the engine.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import secrets
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from .. import __version__
from . import hardware, models, planner
from .supervisor import Settings, Supervisor
from .telemetry import Telemetry

WEBUI = Path(__file__).parent / "webui"


def default_home() -> Path:
    if os.environ.get("SEVENTY_HOME"):
        return Path(os.environ["SEVENTY_HOME"])
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "seventy"


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class Prefs:
    DEFAULTS = {"hf_endpoint": "", "model_dirs": [], "instance_name": "", "autostart": True, "auto_restart": True,
                "last_settings": None}

    def __init__(self, home: Path):
        self.path = home / "prefs.json"
        self.lock = threading.Lock()

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            data = {}
        return {**self.DEFAULTS, **data}

    def update(self, **changes) -> dict:
        with self.lock:
            data = self.load()
            data.update({k: v for k, v in changes.items() if k in self.DEFAULTS})
            self.path.write_text(json.dumps(data, indent=1))
            return data


def create_app(host: str = "127.0.0.1", home: Path | None = None) -> FastAPI:
    home = Path(home or default_home())
    home.mkdir(parents=True, exist_ok=True)
    home.chmod(0o700)
    token_file = home / "ui_token"
    if not token_file.is_file():
        token_file.write_text(secrets.token_urlsafe(24))
        token_file.chmod(0o600)
    token = token_file.read_text().strip()
    need_token = not is_loopback(host)
    prefs = Prefs(home)
    sup = Supervisor(home, gpu_snapshot=lambda: [vars(g) for g in hardware.sample()])
    sup.auto_restart = bool(prefs.load()["auto_restart"])
    profiles = planner.Profiles(home)
    downloads = models.Downloads()
    engine_info: dict = {}

    def engine_url() -> str | None:
        return sup.settings.url if sup.status.state == "ready" and sup.settings else None

    tele = Telemetry(home, engine_url)

    def watch_ready() -> None:
        seen = None
        while True:
            time.sleep(2.0)
            st = sup.status
            if st.state == "ready" and st.run != seen and sup.settings:
                try:
                    req = urllib.request.Request(sup.settings.url + "/info")
                    if sup.settings.api_key:
                        req.add_header("Authorization", f"Bearer {sup.settings.api_key}")
                    with urllib.request.urlopen(req, timeout=5) as r:
                        info = json.load(r)
                    engine_info.clear()
                    engine_info.update(info)
                    mem = info.get("memory") or {}
                    if mem.get("gpu_total") and "budget_min" in mem:
                        util = sup.settings.gpu_memory_utilization
                        overhead = (int(mem["gpu_total"] * util) - int(mem["reserve"]) - int(mem["budget_min"])
                                    - int(mem["weights"]) - int(mem.get("drafter", 0)))
                        profiles.put(sup.settings, {"gpu_total": mem["gpu_total"], "weights": mem["weights"],
                                                    "drafter": mem.get("drafter", 0), "graphs": mem.get("graphs", 0),
                                                    "overhead": max(0, overhead)})
                    seen = st.run
                except Exception:
                    pass
            elif st.state != "ready":
                seen = None if st.state in ("cold", "crashed") else seen

    @asynccontextmanager
    async def lifespan(app):
        tele.start()
        threading.Thread(target=watch_ready, daemon=True, name="ready-watch").start()
        p = prefs.load()
        good = sup.last_good()
        if p["autostart"] and good is not None:
            try:
                sup.start(good)
            except Exception:
                pass
        yield
        tele.stop()
        await asyncio.to_thread(sup.stop)

    app = FastAPI(lifespan=lifespan, title="seventy panel", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if need_token and request.url.path.startswith("/api/"):
            got = request.headers.get("x-seventy-token") or request.query_params.get("token")
            if not got or not secrets.compare_digest(got, token):
                return JSONResponse({"error": "UI token required"}, status_code=401)
        return await call_next(request)

    def settings_from(body: dict) -> Settings:
        try:
            s = Settings.from_dict(body)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"bad settings: {e}")
        if not s.model or not s.gpus:
            raise HTTPException(400, "choose a model and at least one GPU")
        return s

    @app.get("/")
    async def index():
        return FileResponse(WEBUI / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/static/{name}")
    async def static(name: str):
        f = WEBUI / Path(name).name
        if not f.is_file():
            raise HTTPException(404)
        return FileResponse(f, headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    async def state():
        st = sup.status
        p = prefs.load()
        good = sup.last_good()
        return {"status": vars(st), "version": __version__, "instance_name": p["instance_name"],
                "auto_restart": sup.auto_restart, "autostart": p["autostart"],
                "last_good": good.public() if good else None, "last_settings": p["last_settings"],
                "engine": dict(engine_info) if st.state == "ready" else None, "token_required": need_token}

    @app.get("/api/inventory")
    async def inventory():
        return (await asyncio.to_thread(hardware.inventory)).to_dict()

    @app.get("/api/models")
    async def model_list():
        found = await asyncio.to_thread(models.scan, prefs.load()["model_dirs"])
        return [vars(m) for m in found]

    @app.post("/api/models/check")
    async def model_check(body: dict):
        return vars(await asyncio.to_thread(models.check, str(body.get("path", ""))))

    @app.post("/api/plan")
    async def make_plan(body: dict):
        s = settings_from(body)
        inv = await asyncio.to_thread(hardware.inventory)
        return await asyncio.to_thread(planner.plan, s, inv, profiles)

    @app.post("/api/start")
    async def start(body: dict):
        s = settings_from(body)
        inv = await asyncio.to_thread(hardware.inventory)
        p = await asyncio.to_thread(planner.plan, s, inv, profiles)
        if p["errors"]:
            raise HTTPException(400, " ".join(p["errors"]))
        try:
            await asyncio.to_thread(sup.start, s)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        public = dict(body)
        public.pop("api_key", None)
        prefs.update(last_settings=public)
        return {"ok": True, "plan": p}

    @app.post("/api/stop")
    async def stop():
        await asyncio.to_thread(sup.stop)
        return {"ok": True}

    @app.post("/api/restart")
    async def restart():
        s = sup.settings or sup.last_good()
        if s is None:
            raise HTTPException(400, "nothing to restart")
        await asyncio.to_thread(sup.stop)
        await asyncio.to_thread(sup.start, s)
        return {"ok": True}

    @app.post("/api/prefs")
    async def set_prefs(body: dict):
        data = prefs.update(**body)
        sup.auto_restart = bool(data["auto_restart"])
        return data

    @app.get("/api/prefs")
    async def get_prefs():
        return prefs.load()

    @app.post("/api/api_key")
    async def new_api_key():
        return {"api_key": "sk-" + secrets.token_urlsafe(24)}

    @app.get("/api/live")
    async def live():
        return tele.latest() or {}

    @app.get("/api/history")
    async def history(seconds: float = 900, points: int = 600):
        seconds = max(60.0, min(seconds, 30 * 86400))
        return await asyncio.to_thread(tele.history, seconds, max(50, min(points, 2000)))

    @app.get("/api/log")
    async def log(lines: int = 400):
        return PlainTextResponse(await asyncio.to_thread(sup.log_tail, max(10, min(lines, 5000))))

    @app.get("/api/reports")
    async def reports():
        return await asyncio.to_thread(sup.reports)

    @app.get("/api/reports/{name}")
    async def report(name: str):
        r = await asyncio.to_thread(sup.report, name)
        if r is None:
            raise HTTPException(404)
        return r

    @app.get("/api/downloads")
    async def download_list():
        return downloads.list()

    @app.post("/api/downloads")
    async def download(body: dict):
        p = prefs.load()
        endpoint = body.get("endpoint") or p["hf_endpoint"] or os.environ.get("HF_ENDPOINT") or None
        try:
            d = downloads.start(str(body.get("repo", "")), body.get("revision") or None, endpoint,
                                body.get("token") or os.environ.get("HF_TOKEN"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return d.to_dict()

    app.state.token = token
    app.state.need_token = need_token
    app.state.supervisor = sup
    return app


def run(host: str, port: int, home: Path | None = None) -> None:
    import uvicorn

    app = create_app(host, home)
    shown = host if host not in ("0.0.0.0", "::") else _lan_address()
    url = f"http://{shown}:{port}/" + (f"?token={app.state.token}" if app.state.need_token else "")
    print(f"seventy panel {__version__}: {url}", flush=True)
    if app.state.need_token:
        print("seventy panel: open to the network; the UI token is in the URL above", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=10)


def _lan_address() -> str:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
