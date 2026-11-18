"use strict";

const TOKEN_KEY = "seventy-token";
(() => {
  const q = new URLSearchParams(location.search);
  if (q.get("token")) {
    sessionStorage.setItem(TOKEN_KEY, q.get("token"));
    history.replaceState(null, "", location.pathname);
  }
})();

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("aria-") && typeof v === "boolean") { n.setAttribute(k, String(v)); continue; }
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "style") n.style.cssText = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}

function svg(tag, attrs = {}) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

async function api(path, opts = {}) {
  const r = await fetch(path, {
    method: opts.method || (opts.body ? "POST" : "GET"),
    headers: { "Content-Type": "application/json", "X-Seventy-Token": sessionStorage.getItem(TOKEN_KEY) || "" },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = j.detail || j.error || msg; } catch (_) { }
    throw new Error(msg);
  }
  const type = r.headers.get("content-type") || "";
  return type.includes("json") ? r.json() : r.text();
}

const fmt = {
  num(v, d = 0) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(1) + "B";
    if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 1 : 2) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(1) + "K";
    return v.toLocaleString(undefined, { maximumFractionDigits: d });
  },
  int(v) { return v === null || v === undefined ? "–" : Math.round(v).toLocaleString(); },
  gib(b) { return b === null || b === undefined ? "–" : (b / 2 ** 30).toFixed(b >= 10 * 2 ** 30 ? 0 : 1) + " GiB"; },
  pct(v) { return v === null || v === undefined ? "–" : (100 * v).toFixed(v >= 0.995 ? 0 : 1) + "%"; },
  secs(v) {
    if (v === null || v === undefined) return "–";
    if (v < 1) return Math.round(v * 1000) + " ms";
    if (v < 60) return v.toFixed(v < 10 ? 2 : 1) + " s";
    return (v / 60).toFixed(1) + " min";
  },
  dur(s) {
    if (!s && s !== 0) return "–";
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h} h ${m} min` : m ? `${m} min` : `${Math.round(s)} s`;
  },
  time(t, long) {
    const d = new Date(t * 1000);
    const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return long ? d.toLocaleDateString([], { weekday: "short" }) + " " + hm : hm;
  },
};

/* ---------- theme ---------- */
(() => {
  const saved = localStorage.getItem("seventy-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme").addEventListener("click", () => {
    const dark = getComputedStyle(document.body).colorScheme.includes("dark");
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("seventy-theme", next);
    redrawCharts();
  });
})();

/* ---------- views ---------- */
let view = "dashboard";
function show(name) {
  view = name;
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", b.dataset.view === name);
  for (const s of document.querySelectorAll(".view")) s.hidden = s.id !== "view-" + name;
  if (name === "start") loadStart();
  if (name === "models") loadModels();
  if (name === "log") loadLog();
  if (name === "reports") loadReports();
  if (name === "settings") loadSettings();
  if (name === "dashboard") refreshDashboard(true);
}
for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => show(b.dataset.view));

/* ---------- state ---------- */
let state = null;
const STAGES = [
  ["process", "Start the engine process"], ["workers", "Start the tensor-parallel workers"], ["weights", "Load the weights"],
  ["drafter", "Load the DFlash2 drafter"], ["cache", "Allocate the KV cache"], ["graphs", "Capture CUDA graphs"],
  ["server", "Start the API server"], ["ready", "Ready"],
];
const STATE_LOOK = {
  cold: ["■", "Stopped"], starting: ["◌", "Starting"], ready: ["●", "Serving"], stopping: ["◌", "Stopping"], crashed: ["✕", "Crashed"],
};

async function refreshState() {
  try { state = await api("/api/state"); } catch (e) { setChip("crashed", "Panel unreachable"); return; }
  const st = state.status;
  const [ico, label] = STATE_LOOK[st.state] || ["?", st.state];
  setChip(st.state, label, ico);
  $("instance").textContent = state.instance_name ? "· " + state.instance_name : "";
  const s = st.settings;
  $("endpoint").textContent = s && st.state !== "cold" ? `http://${s.host === "0.0.0.0" ? location.hostname : s.host}:${s.port}/v1` : "";
  $("btn-stop").hidden = !["starting", "ready"].includes(st.state);
  $("btn-restart").hidden = !["ready", "crashed", "cold"].includes(st.state) || !(s || state.last_good);
  const banner = $("banner");
  if (st.state === "crashed") {
    banner.hidden = false;
    banner.className = "banner";
    banner.replaceChildren(el("strong", {}, "The engine stopped: "), st.detail || "",
      st.last_report ? el("span", {}, " · ", el("a", { href: "#", onclick: (e) => { e.preventDefault(); show("reports"); openReport(st.last_report); } }, "error report")) : "");
  } else if (st.state === "cold" && view === "dashboard") {
    banner.hidden = false;
    banner.className = "banner info";
    banner.replaceChildren("No engine is running. ", el("a", { href: "#", onclick: (e) => { e.preventDefault(); show("start"); } }, "Start one"),
      state.last_good ? el("span", {}, " or ", el("a", { href: "#", onclick: (e) => { e.preventDefault(); restart(); } }, "start the last good settings")) : "", ".");
  } else banner.hidden = true;
  renderStarting(st);
}

function setChip(kind, label, ico = "✕") {
  const c = $("state");
  c.className = "chip " + kind;
  c.querySelector(".icon").textContent = ico;
  c.querySelector(".label").textContent = label;
}

async function renderStarting(st) {
  const box = $("starting");
  if (st.state !== "starting") { box.hidden = true; return; }
  box.hidden = false;
  const s = st.settings || {};
  const skip = new Set();
  if (!s.draft_model) skip.add("drafter");
  if ((s.gpus || []).length < 2) skip.add("workers");
  if (s.cuda_graphs === false) skip.add("graphs");
  const order = STAGES.filter(([k]) => !skip.has(k));
  const at = order.findIndex(([k]) => k === st.stage);
  $("stages").replaceChildren(...order.map(([k, label], i) => {
    const cls = i < at ? "done" : i === at ? "now" : "";
    return el("li", { class: cls }, el("span", { class: "ico" }, i < at ? "✓" : i === at ? "◌" : "·"), label,
      i === at && st.detail ? el("span", { class: "muted" }, " — " + st.detail) : "");
  }));
  try {
    const text = await api("/api/log?lines=40");
    const pre = $("start-log");
    pre.textContent = text;
    pre.scrollTop = pre.scrollHeight;
  } catch (_) { }
}

$("btn-stop").addEventListener("click", async () => {
  if (!confirm("Stop the engine? Requests in progress end.")) return;
  await api("/api/stop", { method: "POST" }).catch((e) => alert(e.message));
  refreshState();
});
$("btn-cancel").addEventListener("click", async () => { await api("/api/stop", { method: "POST" }).catch(() => { }); refreshState(); });
$("btn-restart").addEventListener("click", () => restart());
async function restart() {
  try { await api("/api/restart", { method: "POST" }); show("dashboard"); } catch (e) { alert(e.message); }
  refreshState();
}

/* ---------- charts ---------- */
const COLORS = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7", "--series-8"];
const css = (v) => getComputedStyle(document.body).getPropertyValue(v).trim();

function niceTicks(max, count = 4) {
  if (!(max > 0)) max = 1;
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
  const top = Math.ceil(max / step) * step;
  const out = [];
  for (let v = 0; v <= top + step / 2; v += step) out.push(+v.toFixed(10));
  return out;
}

class LineChart {
  constructor(host, cfg) {
    this.cfg = cfg;
    this.card = el("div", { class: "chart" });
    this.title = el("h3", {}, cfg.title);
    this.sub = el("div", { class: "sub" }, cfg.sub || "");
    this.legend = el("div", { class: "legend" });
    this.svg = svg("svg", { width: "100%", height: cfg.height || 190, tabindex: 0, role: "img", "aria-label": cfg.title });
    this.table = el("div", { class: "table-wrap" });
    this.card.append(this.title, this.sub, this.legend, this.svg, this.table);
    host.append(this.card);
    this.hover = null;
    this.svg.addEventListener("pointermove", (e) => this.onMove(e));
    this.svg.addEventListener("pointerleave", () => { this.hover = null; this.drawHover(); });
    this.svg.addEventListener("focus", () => { if (this.hover === null && this.times.length) { this.hover = this.times.length - 1; this.drawHover(); } });
    this.svg.addEventListener("blur", () => { this.hover = null; this.drawHover(); });
    this.svg.addEventListener("keydown", (e) => {
      if (!this.times.length) return;
      if (e.key === "ArrowLeft") this.hover = Math.max(0, (this.hover ?? this.times.length) - 1);
      else if (e.key === "ArrowRight") this.hover = Math.min(this.times.length - 1, (this.hover ?? -1) + 1);
      else return;
      e.preventDefault();
      this.drawHover();
    });
    this.times = [];
  }

  update(times, series, range) {
    this.times = times;
    this.series = series;
    this.range = range;
    this.draw();
  }

  draw() {
    const { cfg, svg: s } = this;
    const W = Math.max(280, s.clientWidth || 440), H = cfg.height || 190;
    const m = { l: 50, r: this.series.length > 1 ? 112 : 70, t: 10, b: 22 };
    this.dims = { W, H, m };
    s.setAttribute("viewBox", `0 0 ${W} ${H}`);
    s.replaceChildren();
    const multi = this.series.length > 1;
    this.legend.replaceChildren(...(multi ? this.series.map((ser) =>
      el("span", {}, el("span", { class: "key", style: `background:var(${ser.color})` }), ser.name)) : []));
    const has = this.series.some((ser) => ser.values.some((v) => v !== null && v !== undefined));
    const now = Date.now() / 1000;
    const x0 = now - this.range, x1 = now;
    this.x = (t) => m.l + ((t - x0) / (x1 - x0)) * (W - m.l - m.r);
    let ymax = cfg.yMax ?? 0;
    if (cfg.yMax === undefined) for (const ser of this.series) for (const v of ser.values) if (v > ymax) ymax = v;
    const ticks = niceTicks(ymax || cfg.yFloor || 1);
    const top = ticks[ticks.length - 1];
    this.top = top;
    this.y = (v) => H - m.b - (v / top) * (H - m.t - m.b);
    for (const tv of ticks) {
      const yy = this.y(tv);
      s.append(svg("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, stroke: tv === 0 ? css("--axis") : css("--grid"), "stroke-width": 1, "shape-rendering": "crispEdges" }));
      const t = svg("text", { x: m.l - 6, y: yy + 3.5, "text-anchor": "end" });
      t.textContent = cfg.tick ? cfg.tick(tv) : fmt.num(tv, 2);
      s.append(t);
    }
    const long = this.range > 86400;
    const nx = Math.max(2, Math.min(6, Math.floor((W - m.l - m.r) / 90)));
    for (let i = 0; i <= nx; i++) {
      const t = x0 + ((x1 - x0) * i) / nx;
      const lbl = svg("text", { x: this.x(t), y: H - 6, "text-anchor": i === 0 ? "start" : i === nx ? "end" : "middle" });
      lbl.textContent = fmt.time(t, long);
      s.append(lbl);
    }
    if (!has) {
      const t = svg("text", { x: (m.l + W - m.r) / 2, y: (m.t + H - m.b) / 2, "text-anchor": "middle" });
      t.textContent = cfg.empty || "No data in this range";
      s.append(t);
      this.renderTable();
      return;
    }
    const ends = [];
    this.series.forEach((ser) => {
      const color = css(ser.color);
      let d = "", area = "", open = false, firstX = 0, lastX = 0, last = null;
      ser.values.forEach((v, i) => {
        if (v === null || v === undefined || Number.isNaN(v)) {
          if (open && !multi) area += `L${lastX},${this.y(0)}L${firstX},${this.y(0)}Z`;
          open = false;
          return;
        }
        const px = this.x(this.times[i]), py = this.y(Math.min(v, top));
        if (!open) { d += `M${px},${py}`; if (!multi) area += `M${px},${this.y(0)}L${px},${py}`; firstX = px; open = true; }
        else { d += `L${px},${py}`; if (!multi) area += `L${px},${py}`; }
        lastX = px;
        last = [px, py, v];
      });
      if (open && !multi) area += `L${lastX},${this.y(0)}L${firstX},${this.y(0)}Z`;
      if (!multi && area) s.append(svg("path", { d: area, fill: color, "fill-opacity": 0.1, stroke: "none" }));
      s.append(svg("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      if (last) ends.push({ ser, color, x: last[0], y: last[1], v: last[2] });
    });
    ends.sort((a, b) => a.y - b.y);
    let prevY = -1e9;
    for (const e of ends) {
      s.append(svg("circle", { cx: e.x, cy: e.y, r: 4, fill: e.color, stroke: css("--surface-1"), "stroke-width": 2 }));
      if (e.y - prevY < 13) continue;
      prevY = e.y;
      const t = svg("text", { x: e.x + 8, y: e.y + 4, class: "end-label" });
      t.textContent = (multi ? e.ser.name + " " : "") + (cfg.fmt || fmt.num)(e.v);
      s.append(t);
    }
    this.hoverLayer = svg("g");
    s.append(this.hoverLayer);
    this.drawHover();
    this.renderTable();
  }

  onMove(e) {
    if (!this.times.length || !this.dims) return;
    const r = this.svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * this.dims.W;
    let best = 0, bd = Infinity;
    this.times.forEach((t, i) => { const d = Math.abs(this.x(t) - px); if (d < bd) { bd = d; best = i; } });
    this.hover = best;
    this.pointer = [e.clientX, e.clientY];
    this.drawHover();
  }

  drawHover() {
    const tip = $("tooltip");
    if (!this.hoverLayer) return;
    this.hoverLayer.replaceChildren();
    if (this.hover === null || this.hover >= this.times.length) { if (tip.owner === this) tip.hidden = true; return; }
    const i = this.hover, { H, m } = this.dims, x = this.x(this.times[i]);
    this.hoverLayer.append(svg("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1 }));
    const rows = [];
    for (const ser of this.series) {
      const v = ser.values[i];
      if (v !== null && v !== undefined) this.hoverLayer.append(svg("circle", { cx: x, cy: this.y(Math.min(v, this.top)), r: 4, fill: css(ser.color), stroke: css("--surface-1"), "stroke-width": 2 }));
      rows.push(el("div", { class: "tt-row" }, el("span", { class: "key", style: `background:var(${ser.color})` }),
        el("strong", {}, v === null || v === undefined ? "–" : (this.cfg.fmt || fmt.num)(v)), el("span", { class: "name" }, ser.name)));
    }
    tip.replaceChildren(el("div", { class: "tt-time" }, new Date(this.times[i] * 1000).toLocaleString()), ...rows);
    tip.hidden = false;
    tip.owner = this;
    const r = this.svg.getBoundingClientRect();
    const cx = r.left + (x / this.dims.W) * r.width;
    const w = tip.offsetWidth || 160;
    tip.style.left = (cx + 14 + w > innerWidth ? cx - w - 14 : cx + 14) + "px";
    tip.style.top = (r.top + 10) + "px";
  }

  renderTable() {
    if (!$("show-tables").checked || !this.times.length) { this.table.replaceChildren(); return; }
    const head = el("tr", {}, el("th", {}, "Time"), ...this.series.map((s) => el("th", {}, s.name + (this.cfg.unit ? ` (${this.cfg.unit})` : ""))));
    const body = [];
    for (let i = this.times.length - 1; i >= 0; i--) {
      body.push(el("tr", {}, el("td", {}, new Date(this.times[i] * 1000).toLocaleTimeString()),
        ...this.series.map((s) => el("td", {}, s.values[i] === null || s.values[i] === undefined ? "–" : (this.cfg.fmt || fmt.num)(s.values[i])))));
    }
    this.table.replaceChildren(el("table", { class: "data" }, el("thead", {}, head), el("tbody", {}, ...body)));
  }
}

const RANGES = [["5 min", 300], ["15 min", 900], ["1 h", 3600], ["6 h", 21600], ["24 h", 86400], ["7 days", 604800]];
let range = +(localStorage.getItem("seventy-range") || 900);
$("ranges").replaceChildren(...RANGES.map(([label, secs]) => el("button", {
  "aria-pressed": secs === range, onclick: () => {
    range = secs; localStorage.setItem("seventy-range", secs);
    for (const b of $("ranges").children) b.setAttribute("aria-pressed", b.textContent.replace("✓ ", "") === label);
    refreshDashboard(true);
  },
}, label)));
$("show-tables").addEventListener("change", () => redrawCharts());

const charts = {};
let hist = [];
const eng = (k) => hist.map((h) => (h.engine ? h.engine[k] ?? null : null));
const gpuSeries = (k, scale = 1) => {
  const n = Math.max(0, ...hist.map((h) => (h.gpus || []).length));
  return Array.from({ length: n }, (_, g) => ({
    name: "GPU " + g, color: COLORS[g % 8],
    values: hist.map((h) => (h.gpus && h.gpus[g] && h.gpus[g][k] !== null && h.gpus[g][k] !== undefined ? h.gpus[g][k] * scale : null)),
  }));
};
const CHARTS = [
  ["gen", { title: "Generated tokens per second", sub: "all streams together", unit: "tok/s" },
    () => [{ name: "Generated", color: COLORS[0], values: eng("gen_tps") }]],
  ["prompt", { title: "Prompt tokens per second", sub: "prompt tokens computed (prefix-cache hits excluded)", unit: "tok/s" },
    () => [{ name: "Prompt", color: COLORS[0], values: eng("prompt_tps") }]],
  ["ttft", { title: "Time to first token", sub: "median (p50) and 95th percentile (p95) of the requests in each interval", unit: "s", fmt: fmt.secs, tick: fmt.secs },
    () => [{ name: "p50", color: COLORS[0], values: eng("ttft_p50") }, { name: "p95", color: COLORS[1], values: eng("ttft_p95") }]],
  ["reqs", { title: "Requests", sub: "in the running batch and waiting for a slot", fmt: fmt.int },
    () => [{ name: "Running", color: COLORS[0], values: eng("running") }, { name: "Waiting", color: COLORS[1], values: eng("waiting") }]],
  ["kv", { title: "KV cache in use", sub: "share of the KV cache blocks", yMax: 1, fmt: fmt.pct, tick: (v) => Math.round(v * 100) + "%" },
    () => [{ name: "KV cache", color: COLORS[0], values: eng("kv_usage") }]],
  ["spec", { title: "Tokens per verify step", sub: "DFlash2: accepted drafts plus the target's own token", fmt: (v) => v.toFixed(2) },
    () => [{ name: "Tokens per step", color: COLORS[0], values: eng("tokens_per_step") }]],
  ["cache", { title: "Prompt tokens from the prefix cache", sub: "share of prompt tokens not computed again", yMax: 1, fmt: fmt.pct, tick: (v) => Math.round(v * 100) + "%" },
    () => [{ name: "Reused", color: COLORS[0], values: eng("cache_hit") }]],
  ["gutil", { title: "GPU utilization", yMax: 100, fmt: (v) => Math.round(v) + "%", tick: (v) => v + "%" }, () => gpuSeries("util")],
  ["gpow", { title: "GPU power", fmt: (v) => Math.round(v) + " W", tick: (v) => v + " W" }, () => gpuSeries("power")],
  ["gmem", { title: "GPU memory used", fmt: (v) => v.toFixed(1) + " GiB", tick: (v) => +v.toFixed(2) + " GiB" }, () => gpuSeries("memory_used", 1 / 2 ** 30)],
  ["gtemp", { title: "GPU temperature", fmt: (v) => Math.round(v) + " °C", tick: (v) => v + " °C" }, () => gpuSeries("temperature")],
  ["cpu", { title: "Host CPU", yMax: 100, fmt: (v) => Math.round(v) + "%", tick: (v) => v + "%" },
    () => [{ name: "CPU", color: COLORS[0], values: hist.map((h) => h.cpu ?? null) }]],
];

function redrawCharts() {
  const times = hist.map((h) => h.t);
  for (const [key, cfg, fn] of CHARTS) {
    if (!charts[key]) charts[key] = new LineChart($("charts"), cfg);
    charts[key].update(times, fn(), range);
  }
  renderTiles();
}

let lastHistory = 0;
async function refreshDashboard(force) {
  if (view !== "dashboard") return;
  const now = Date.now();
  if (!force && range > 3600 && now - lastHistory < 30000) return;
  lastHistory = now;
  for (const c of Object.values(charts)) c.card.classList.add("loading");
  try { hist = await api(`/api/history?seconds=${range}&points=${range <= 900 ? 450 : 600}`); } catch (_) { }
  for (const c of Object.values(charts)) c.card.classList.remove("loading");
  redrawCharts();
}

function renderTiles() {
  const last = hist.length ? hist[hist.length - 1] : null;
  const e = last && last.engine;
  const gpus = (last && last.gpus) || [];
  const power = gpus.reduce((a, g) => a + (g.power || 0), 0);
  const info = state && state.engine;
  const restarts = state && state.status.restarts ? state.status.restarts.length : 0;
  const tiles = [
    ["Tokens per second", e ? fmt.num(e.gen_tps, 1) : "–", e ? `generated · prompt ${fmt.num(e.prompt_tps, 0)}` : "no engine"],
    ["Requests running", e ? fmt.int(e.running) : "–", e ? `${fmt.int(e.waiting)} waiting` : ""],
    ["Time to first token", e && e.ttft_p50 !== null && e.ttft_p50 !== undefined ? fmt.secs(e.ttft_p50) : "–", e && e.ttft_p95 ? `p95 ${fmt.secs(e.ttft_p95)}` : "p50, last interval"],
    ["KV cache in use", e ? fmt.pct(e.kv_usage) : "–", info ? `${fmt.num(info.kv_tokens)} tokens per GPU` : ""],
    ["Tokens per step", e && e.tokens_per_step ? e.tokens_per_step.toFixed(2) : "–", info && info.draft ? "DFlash2 verify" : "DFlash2 off"],
    ["Prefix cache reuse", e && e.cache_hit !== null && e.cache_hit !== undefined ? fmt.pct(e.cache_hit) : "–", "of prompt tokens"],
    ["GPU power", gpus.length ? Math.round(power) + " W" : "–", gpus.length ? `${gpus.length} GPUs` : ""],
    ["Uptime", state && state.status.ready_at ? fmt.dur(Date.now() / 1000 - state.status.ready_at) : "–",
      restarts ? `${restarts} automatic restart${restarts === 1 ? "" : "s"}` : ""],
  ];
  $("tiles").replaceChildren(...tiles.map(([l, v, s]) => el("div", { class: "tile" }, el("div", { class: "t-label" }, l), el("div", { class: "t-value" }, v), el("div", { class: "t-sub" }, s))));
}
addEventListener("resize", () => { clearTimeout(window._rz); window._rz = setTimeout(redrawCharts, 150); });

/* ---------- start ---------- */
let inv = null, modelList = [], planTimer = null;
const field = {
  model: "f-model", draft_model: "f-draft", max_model_len: "f-maxlen", max_num_seqs: "f-seqs", gpu_memory_utilization: "f-util",
  prefill_budget: "f-prefill", kv_cache_dtype: "f-kv", max_images_per_request: "f-images", max_videos_per_request: "f-videos",
  max_image_pixels: "f-pixels", served_model_name: "f-name", host: "f-host", port: "f-port", api_key: "f-key",
};

async function loadStart() {
  try {
    [inv, modelList] = await Promise.all([api("/api/inventory"), api("/api/models")]);
  } catch (e) { $("machine").textContent = e.message; return; }
  renderMachine();
  const targets = modelList.filter((m) => m.kind === "target");
  const drafters = modelList.filter((m) => m.kind === "drafter");
  $("f-model").replaceChildren(...(targets.length ? targets : [{ path: "", name: "No models found (see Models)", ok: false }]).map((m) =>
    el("option", { value: m.path, disabled: !m.ok || !m.path }, `${m.name}${m.size ? " (" + fmt.gib(m.size) + ")" : ""}${m.ok ? "" : " — cannot serve"}`)));
  $("f-draft").replaceChildren(el("option", { value: "" }, "None"), ...drafters.map((m) =>
    el("option", { value: m.path, disabled: !m.ok }, `${m.name} (${fmt.gib(m.size)})`)));
  $("f-gpus").replaceChildren(...inv.gpus.map((g) => el("label", {},
    el("input", { type: "checkbox", value: g.id, checked: true, onchange: schedulePlan }),
    el("span", { class: "gname" }, `GPU ${g.id}`), el("span", {}, `${g.name} · ${fmt.gib(g.memory_total)}`),
    el("span", { class: "muted" }, `NVLink to ${Object.keys(g.links).length} GPU(s)`))));
  const s = (state && (state.last_settings || state.last_good)) || {};
  if (s.model && targets.some((m) => m.path === s.model)) $("f-model").value = s.model;
  else { const ok = targets.find((m) => m.ok); if (ok) $("f-model").value = ok.path; }
  $("f-draft").value = s.draft_model && drafters.some((m) => m.path === s.draft_model) ? s.draft_model : (drafters.find((m) => m.ok) || { path: "" }).path;
  if (s.gpus) for (const c of $("f-gpus").querySelectorAll("input")) c.checked = s.gpus.includes(+c.value);
  for (const [k, id] of Object.entries(field)) if (k in s && s[k] !== null && k !== "model" && k !== "draft_model" && k !== "api_key") $(id).value = s[k];
  if ("thinking" in s) $("f-think").value = s.thinking ? "1" : "0";
  if ("cuda_graphs" in s) $("f-graphs").checked = s.cuda_graphs;
  if ("prefix_cache" in s) $("f-prefix").checked = s.prefix_cache;
  modelNote();
  schedulePlan();
}

function renderMachine() {
  if (inv.error) { $("machine").replaceChildren(el("div", { class: "error" }, inv.error)); return; }
  const dl = el("dl", { class: "machine-grid" },
    el("dt", {}, "GPUs"), el("dd", {}, inv.gpus.length ? `${inv.gpus.length} × ${inv.gpus[0].name}` : "none found"),
    el("dt", {}, "Driver"), el("dd", {}, inv.driver),
    el("dt", {}, "CPU"), el("dd", {}, `${inv.cpu} (${inv.cpu_threads} threads)`),
    el("dt", {}, "RAM"), el("dd", {}, `${fmt.gib(inv.ram_available)} free of ${fmt.gib(inv.ram_total)}`),
    el("dt", {}, "Disk"), el("dd", {}, `${fmt.gib(inv.disk_free)} free at `, el("span", { class: "mono" }, inv.disk_path)));
  const ids = inv.gpus.map((g) => g.id);
  const links = ids.length > 1 ? el("table", { class: "links" },
    el("tr", {}, el("th", {}, "NVLink"), ...ids.map((i) => el("th", {}, "GPU " + i))),
    ...inv.gpus.map((g) => el("tr", {}, el("th", {}, "GPU " + g.id), ...ids.map((j) => el("td", {}, j === g.id ? "–" : String(g.links[j] || 0)))))) : "";
  $("machine").replaceChildren(dl, links, ids.length > 1 ? el("div", { class: "note" }, "Links between each pair of GPUs.") : "");
}

function modelNote() {
  const m = modelList.find((x) => x.path === $("f-model").value);
  const n = $("model-note");
  if (!m) { n.textContent = ""; return; }
  const d = m.details || {};
  n.textContent = m.ok ? `${d.architecture || ""} · ${d.layers} layers · weights ${d.weights} · ${d.vision ? "images and video" : "text only"} · up to ${fmt.int(d.max_context)} tokens of context`
    : "Cannot serve: " + (m.reasons || []).join("; ");
  if (!$("f-maxlen").value && d.max_context) $("f-maxlen").value = d.max_context;
}

function settingsFromForm() {
  const num = (id) => ($(id).value === "" ? null : +$(id).value);
  return {
    model: $("f-model").value, draft_model: $("f-draft").value || null,
    gpus: [...$("f-gpus").querySelectorAll("input:checked")].map((c) => +c.value),
    max_model_len: num("f-maxlen"), max_num_seqs: num("f-seqs") || 8, gpu_memory_utilization: num("f-util") || 0.9,
    prefill_budget: num("f-prefill") || 4096, kv_cache_dtype: $("f-kv").value, thinking: $("f-think").value === "1",
    max_images_per_request: num("f-images") ?? 20, max_videos_per_request: num("f-videos") ?? 1, max_image_pixels: num("f-pixels"),
    served_model_name: $("f-name").value || null, host: $("f-host").value, port: num("f-port") || 8000,
    api_key: $("f-key").value || null, cuda_graphs: $("f-graphs").checked, prefix_cache: $("f-prefix").checked,
  };
}

function schedulePlan() { clearTimeout(planTimer); planTimer = setTimeout(runPlan, 250); }
for (const id of Object.values(field).concat(["f-think", "f-graphs", "f-prefix"])) $(id).addEventListener("input", schedulePlan);
$("f-model").addEventListener("change", () => { $("f-maxlen").value = ""; modelNote(); schedulePlan(); });
$("btn-key").addEventListener("click", async () => { const r = await api("/api/api_key", { method: "POST" }); $("f-key").value = r.api_key; $("f-key").type = "text"; });
$("btn-showkey").addEventListener("click", () => { $("f-key").type = $("f-key").type === "password" ? "text" : "password"; });

async function runPlan() {
  const s = settingsFromForm();
  const box = $("plan");
  if (!s.model) { box.replaceChildren(el("div", { class: "note" }, "Choose a model.")); return; }
  let p;
  try { p = await api("/api/plan", { body: s }); } catch (e) { box.replaceChildren(el("div", { class: "error" }, e.message)); return; }
  const issues = el("ul", { class: "warn-list" },
    ...p.errors.map((t) => el("li", { class: "e" }, el("span", { class: "ico" }, "✕"), el("span", {}, t))),
    ...p.warnings.map((t) => el("li", { class: "w" }, el("span", { class: "ico" }, "▲"), el("span", {}, t))));
  $("btn-start").disabled = !!p.errors.length || !(state && ["cold", "crashed"].includes(state.status.state));
  if (p.errors.length || !p.gpu_bytes) { box.replaceChildren(issues); return; }
  const parts = [
    ["Weights", p.weights, "--series-1", false], ["CUDA and buffers", p.overhead, "--series-2", false],
    ["Activation reserve", p.reserve, "--series-3", false], ["Request state", p.state, "--series-4", true],
    ["KV cache", p.kv, "--series-5", false], ["Outside the budget", p.gpu_bytes - p.budget, "--neutral", "neutral"],
  ].filter(([, v]) => v > 0);
  const total = p.gpu_bytes;
  const bar = el("div", { class: "segbar", role: "img", "aria-label": "GPU memory: " + parts.map(([n, v]) => `${n} ${fmt.gib(v)}`).join(", ") });
  for (const [name, v, color, ink] of parts) {
    const share = v / total;
    const label = fmt.gib(v);
    const part = el("div", { class: "part" + (ink === true ? " dark-ink" : ""), style: `flex:${share} 1 0;background:var(${color})${ink === "neutral" ? ";color:var(--text-primary)" : ""}`, title: `${name}: ${label} (${Math.round(share * 100)}%)` });
    if (share * 360 > label.length * 7 + 12) part.textContent = label;
    part.addEventListener("pointermove", (e) => {
      const tip = $("tooltip");
      tip.replaceChildren(el("div", { class: "tt-row" }, el("strong", {}, label), el("span", { class: "name" }, `${name} · ${Math.round(share * 100)}%`)));
      tip.hidden = false; tip.owner = part;
      tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY + 12 + "px";
    });
    part.addEventListener("pointerleave", () => { $("tooltip").hidden = true; });
    bar.append(part);
  }
  const legend = el("div", { class: "legend" }, ...parts.map(([n, , c]) => el("span", {}, el("span", { class: "box", style: `background:var(${c})` }), n)));
  box.replaceChildren(
    el("div", { class: "plan-kv" }, fmt.int(p.kv_tokens), el("span", { class: "muted", style: "font-size:14px;font-weight:400" }, " KV cache tokens")),
    el("div", { class: "note" }, `${p.tp} GPU(s) · up to ${fmt.int(p.max_model_len)} tokens per request · ${p.measured ? "measured on an earlier start" : "estimated from the checkpoint sizes"}` +
      (p.graphs ? ` · CUDA graphs use ${fmt.gib(p.graphs)} of the activation reserve` : "")),
    el("div", { class: "muted", style: "font-size:13px" }, `Memory of one GPU (${fmt.gib(total)})`), bar, legend, issues);
}

$("btn-start").addEventListener("click", async () => {
  $("start-error").textContent = "";
  $("btn-start").disabled = true;
  try {
    await api("/api/start", { body: settingsFromForm() });
    show("dashboard");
    refreshState();
  } catch (e) { $("start-error").textContent = e.message; $("btn-start").disabled = false; }
});

/* ---------- models ---------- */
async function loadModels() {
  const [list, prefs] = await Promise.all([api("/api/models"), api("/api/prefs")]);
  $("m-endpoint").textContent = prefs.hf_endpoint || "https://huggingface.co (or HF_ENDPOINT)";
  $("models-table").replaceChildren(list.length ? el("table", { class: "list" },
    el("tr", {}, el("th", {}, "Model"), el("th", {}, "Kind"), el("th", {}, "Size"), el("th", {}, "Status"), el("th", {}, "Folder")),
    ...list.map((m) => el("tr", {}, el("td", {}, m.name), el("td", {}, m.kind), el("td", { class: "num" }, fmt.gib(m.size)),
      el("td", {}, m.ok ? el("span", { class: "badge ok" }, el("span", { class: "ico" }, "✓"), "ready") : el("span", { class: "badge bad" }, el("span", { class: "ico" }, "✕"), (m.reasons || []).join("; "))),
      el("td", { class: "mono" }, m.path)))) : el("div", { class: "note" }, "No checkpoints found in the Hugging Face cache or the added folders."));
  renderDownloads();
}

$("btn-adddir").addEventListener("click", async () => {
  const dir = $("m-dir").value.trim();
  if (!dir) return;
  const prefs = await api("/api/prefs");
  await api("/api/prefs", { body: { model_dirs: [...new Set([...(prefs.model_dirs || []), dir])] } });
  $("m-dir").value = "";
  loadModels();
});
$("btn-download").addEventListener("click", async () => {
  try { await api("/api/downloads", { body: { repo: $("m-repo").value } }); $("m-repo").value = ""; renderDownloads(); } catch (e) { alert(e.message); }
});
async function renderDownloads() {
  const list = await api("/api/downloads").catch(() => []);
  $("downloads").replaceChildren(...list.map((d) => el("div", { class: "card", style: "padding:10px;margin:8px 0 0" },
    el("div", { class: "row between" }, el("strong", {}, d.repo), el("span", { class: "muted" }, d.state === "failed" ? "failed" : `${fmt.gib(d.done)} of ${d.total ? fmt.gib(d.total) : "?"}`)),
    d.total ? el("div", { class: "meter" }, el("div", { style: `width:${Math.min(100, (100 * d.done) / d.total)}%` })) : "",
    d.error ? el("div", { class: "error" }, d.error) : "")));
}

/* ---------- log, reports, settings ---------- */
async function loadLog() {
  const text = await api("/api/log?lines=1000").catch((e) => e.message);
  const pre = $("log");
  const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
  pre.textContent = text || "No engine log yet.";
  if ($("log-follow").checked && atEnd) pre.scrollTop = pre.scrollHeight;
}
async function loadReports() {
  const list = await api("/api/reports").catch(() => []);
  $("reports").replaceChildren(list.length ? el("table", { class: "list" },
    el("tr", {}, el("th", {}, "Time"), el("th", {}, "Reason"), el("th", {}, "Stage")),
    ...list.map((r) => el("tr", {}, el("td", {}, el("a", { href: "#", onclick: (e) => { e.preventDefault(); openReport(r.name); } }, new Date(r.time * 1000).toLocaleString())),
      el("td", {}, r.reason || ""), el("td", {}, r.stage || "")))) : el("div", { class: "note" }, "No error reports."));
}
async function openReport(name) {
  const r = await api("/api/reports/" + encodeURIComponent(name));
  $("report-card").hidden = false;
  $("report-title").textContent = r.reason || name;
  $("report-meta").textContent = `${new Date(r.time * 1000).toLocaleString()} · run ${r.run} · stage ${r.stage || "?"} · exit code ${r.exit_code ?? "?"}`;
  $("report-log").textContent = r.log_tail || "";
}
async function loadSettings() {
  const p = await api("/api/prefs");
  $("s-name").value = p.instance_name || "";
  $("s-endpoint").value = p.hf_endpoint || "";
  $("s-autostart").checked = !!p.autostart;
  $("s-restart").checked = !!p.auto_restart;
  $("about").textContent = `seventy ${state ? state.version : ""} · control panel${state && state.token_required ? " · open to the network (UI token required)" : " · this machine only"}`;
}
$("btn-save").addEventListener("click", async () => {
  await api("/api/prefs", { body: { instance_name: $("s-name").value, hf_endpoint: $("s-endpoint").value.trim(), autostart: $("s-autostart").checked, auto_restart: $("s-restart").checked } });
  $("s-saved").textContent = "Saved.";
  setTimeout(() => ($("s-saved").textContent = ""), 2000);
  refreshState();
});

/* ---------- loops ---------- */
async function tick() {
  await refreshState();
  if (view === "dashboard") refreshDashboard(false);
  if (view === "log" && $("log-follow").checked) loadLog();
  if (view === "models") renderDownloads();
}
tick();
setInterval(tick, 2000);
