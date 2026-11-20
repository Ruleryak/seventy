"""Speed charts and a speed sheet from benchmark results, one results file per release.

    python bench/report.py --images docs/images
    python bench/report.py --html speed-sheet.html
    python bench/report.py --markdown

Each file in bench/results/ (or the folder that --results gives) holds the speed of one release:

    {"format": 1, "release": "1.0.0", "name": "1.0.0", "note": "...", "commit": "079485e",
     "date": "2026-11-19T09:15Z", "setup": {...},
     "decode": [{"context": 2000, "streams": 4, "per_stream": [224.2, 154.8], "total": [896.6, 619.3],
                 "ttft_cached": [0.141, 0.134]}, ...],
     "prefill": [{"context": 2000, "tokens_per_s": [2796, 2846]}, ...]}

decode: requests of `context` prompt tokens that are already in the prefix cache, `streams` of them at the same time.
per_stream is the output tokens/s of one request (first to last token), total the output tokens/s of all requests
together while all of them run; each list has one value per run. prefill: prompt tokens/s of one cold prompt.

--markdown prints the tables of the newest release for the README. --images writes the README charts of the newest release (sorted by date), light and dark, with the release before
it as lighter lines. --html writes one self-contained page with every release; each release is drawn over the one
before it.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path

CONTEXT_MAX = 262144
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
SLOTS = 8
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#6f6e69", "grid": "#e6e6e3",
              "axis": "#c3c2b7", "zone": "#f2f4fa", "edge": "#e1e0d9",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]},
    "dark": {"surface": "#0a0d18", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#9a998f", "grid": "#1d2130",
             "axis": "#3a3f4e", "zone": "#121729", "edge": "#2a2f3d",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]},
}
THEMES["css"] = {k: (f"var(--c-{k})" if k != "series" else [f"var(--c-s{i + 1})" for i in range(SLOTS)])
                 for k in THEMES["light"]}
CHARTS = {
    "total": ("Total decode speed", "Output tokens per second of all streams together"),
    "stream": ("Decode speed per stream", "Output tokens per second that each request receives"),
    "prefill": ("Prefill speed", "Prompt tokens per second for one prompt that is not in the cache"),
}


def load(folder: Path) -> list[dict]:
    """All results files of a folder, oldest release first."""
    runs = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    return sorted(runs, key=lambda r: r["date"])


def k_label(tokens: int) -> str:
    """Tick label in units of 1024 tokens (context windows are powers of two: 256K = 262,144)."""
    return "0" if tokens == 0 else f"{tokens // 1024}K"


def series(run: dict, kind: str) -> dict:
    """{streams: [(context, mean, low, high, runs)]}; prefill has one series under the key 0."""
    out: dict[int, list] = {}
    if kind == "prefill":
        for c in run.get("prefill", []):
            v = c["tokens_per_s"]
            out.setdefault(0, []).append((c["context"], statistics.mean(v), min(v), max(v), len(v)))
    else:
        key = "total" if kind == "total" else "per_stream"
        for c in run.get("decode", []):
            v = c[key]
            out.setdefault(c["streams"], []).append((c["context"], statistics.mean(v), min(v), max(v), len(v)))
    return {s: sorted(p) for s, p in sorted(out.items())}


def axis_top(vmax: float) -> tuple[float, float]:
    """A round step for about 5 gridlines and the first multiple of it above vmax."""
    raw = max(vmax, 1.0) / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next(f * mag for f in (1, 2, 2.5, 5, 10) if f * mag >= raw)
    return step, math.ceil(vmax * 1.04 / step) * step


def stream_name(s: int) -> str:
    return "1 stream" if s == 1 else f"{s} streams"


def text(x, y, s, pal, size=13, fill="ink2", anchor="start", weight=400, halo=False) -> str:
    """One SVG text element in the chart font; halo draws a surface outline under the glyphs."""
    extra = (f' paint-order="stroke" stroke="{pal["surface"]}" stroke-width="4" stroke-linejoin="round"'
             if halo else "")
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{pal[fill]}" '
            f'text-anchor="{anchor}"{extra}>{html.escape(s)}</text>')


def chart(kind: str, run: dict, prev: dict | None, theme: str, width: int, height: int, titled: bool) -> str:
    """One chart as an SVG document: the release's points and lines, the previous release as lighter lines, the
    range of the runs as a vertical bar at each point, and the context lengths that were not measured as a band."""
    pal = THEMES[theme]
    narrow = width < 520
    cur, old = series(run, kind), series(prev, kind) if prev else {}
    values = [p[3] for pts in list(cur.values()) + list(old.values()) for p in pts]
    step, top_v = axis_top(max(values) if values else 1.0)
    measured = max((p[0] for pts in cur.values() for p in pts), default=0)
    labels = max((7.6 * len(stream_name(k)) + 22 for k in cur), default=0) if len(cur) > 1 else 0
    left, right, bottom = (46 if narrow else 58), 16 + (labels if measured > CONTEXT_MAX * 0.8 else 0), 48
    names = [stream_name(k) for k in cur] if len(cur) > 1 else []
    lighter = f"Lighter lines: {prev['name']}" if prev and not narrow else ""
    row_end = left + sum(24 + 7.4 * len(n) + 18 for n in names)
    wrap = bool(lighter) and bool(names) and row_end + 24 + 7.4 * len(lighter) > width - right
    top = (104 if titled else 44) + (20 if wrap else 0)
    pw, ph = width - left - right, height - top - bottom
    leg_y = top - 22

    def px(c):
        return left + c / CONTEXT_MAX * pw

    def py(v):
        return top + ph - v / top_v * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
           f'height="{height}" font-family="{FONT}" role="img" aria-label="{html.escape(CHARTS[kind][0])}">']
    out.append(f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" fill="{pal["surface"]}" '
               f'stroke="{pal["edge"]}"/>')
    if titled:
        name = f"seventy {run['name']}" + (f" ({run['note']})" if run.get("note") else "")
        out.append(text(left - 34 if not narrow else 16, 34, CHARTS[kind][0], pal, 19, "ink", weight=600))
        out.append(text(left - 34 if not narrow else 16, 56, CHARTS[kind][1], pal, 14, "ink2"))
        out.append(text(width - right - 4, 34, name, pal, 13, "muted", "end"))
    if measured < CONTEXT_MAX * 0.95:
        zx = px(measured) + 18
        out.append(f'<rect x="{zx:.1f}" y="{top}" width="{left + pw - zx:.1f}" height="{ph}" fill="{pal["zone"]}"/>')
        out.append(text(left + pw - 10, top + 20, "Not measured yet", pal, 13, "muted", "end"))
    v = 0.0
    while v <= top_v + 1e-9:
        y = py(v)
        out.append(f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" '
                   f'stroke="{pal["axis"] if v == 0 else pal["grid"]}" stroke-width="1"/>')
        out.append(text(left - 8, y + 4.5, f"{v:,.0f}", pal, 13, "muted", "end"))
        v += step
    tick = 65536 if pw < 440 else 32768
    for c in range(0, CONTEXT_MAX + 1, tick):
        anchor = "end" if c == CONTEXT_MAX else ("start" if c == 0 else "middle")
        out.append(text(px(c), top + ph + 20, k_label(c), pal, 13, "muted", anchor))
    out.append(text(left + pw / 2, top + ph + 40, "Context length (tokens)", pal, 13, "muted", "middle"))

    keys = list(cur) or list(old)
    color = {s: pal["series"][i % SLOTS] for i, s in enumerate(keys)}
    for s, pts in old.items():
        col = color.get(s, pal["muted"])
        d = " ".join(f"{px(c):.1f},{py(m):.1f}" for c, m, *_ in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-opacity="0.38" stroke-width="1.5" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        for c, m, *_ in pts:
            out.append(f'<circle cx="{px(c):.1f}" cy="{py(m):.1f}" r="2.5" fill="{col}" fill-opacity="0.38"/>')
    hits = []
    for s, pts in cur.items():
        col = color[s]
        d = " ".join(f"{px(c):.1f},{py(m):.1f}" for c, m, *_ in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" '
                   f'stroke-linecap="round"/>')
        before = {p[0]: p[1] for p in old.get(s, [])}
        for c, m, lo, hi, n in pts:
            x, y = px(c), py(m)
            if hi - lo > 0.005 * top_v:
                out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{py(lo):.1f}" y2="{py(hi):.1f}" stroke="{col}" '
                           f'stroke-opacity="0.45" stroke-width="2" stroke-linecap="round"/>')
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}" stroke="{pal["surface"]}" '
                       f'stroke-width="2"/>')
            if theme == "css":
                who = "One prompt" if kind == "prefill" else stream_name(s)
                unit = "prompt tokens/s" if kind == "prefill" else ("tok/s in total" if kind == "total" else
                                                                     "tok/s per stream")
                lines = [f"{who} at {c:,} tokens", f"{m:,.0f} {unit}"]
                if n > 1:
                    lines.append(f"{n} runs: {lo:,.0f} to {hi:,.0f}")
                if c in before and prev:
                    lines.append(f"{prev['name']}: {before[c]:,.0f} ({(m / before[c] - 1) * 100:+.0f}%)")
                hits.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="transparent" tabindex="0" '
                            f'data-tip="{html.escape("|".join(lines))}"/>')
    if len(cur) > 1:
        ends = sorted(((py(pts[-1][1]), px(pts[-1][0]), s) for s, pts in cur.items()), key=lambda e: e[0])
        ys = [e[0] for e in ends]
        for _ in range(50):
            for i in range(1, len(ys)):
                if ys[i] - ys[i - 1] < 17:
                    shift = (17 - (ys[i] - ys[i - 1])) / 2
                    ys[i - 1] -= shift
                    ys[i] += shift
        for (y, x, s), ly in zip(ends, ys):
            lx = x + 16
            if abs(ly - y) > 2:
                out.append(f'<line x1="{x + 6:.1f}" y1="{y:.1f}" x2="{lx - 3:.1f}" y2="{ly:.1f}" '
                           f'stroke="{pal["muted"]}" stroke-width="1"/>')
            out.append(text(lx, ly + 4.5, stream_name(s), pal, 13, "ink", weight=600, halo=True))
    lx = left
    if names:
        if wrap:
            leg_y -= 20
        for s in cur:
            out.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{leg_y}" y2="{leg_y}" stroke="{color[s]}" '
                       f'stroke-width="2"/><circle cx="{lx + 9}" cy="{leg_y}" r="4" fill="{color[s]}"/>')
            out.append(text(lx + 24, leg_y + 4.5, stream_name(s), pal, 13, "ink2"))
            lx += 24 + 7.4 * len(stream_name(s)) + 18
    if lighter:
        if wrap:
            lx, leg_y = left, leg_y + 20
        first = pal["series"][0]
        out.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{leg_y}" y2="{leg_y}" stroke="{first}" '
                   f'stroke-opacity="0.38" stroke-width="1.5"/>')
        out.append(text(lx + 24, leg_y + 4.5, lighter, pal, 13, "ink2"))
    out += hits
    out.append("</svg>")
    return "\n".join(out)


def write_images(runs: list[dict], folder: Path) -> list[Path]:
    """README charts of the newest release, light and dark: speed-<chart>-<theme>.svg."""
    folder.mkdir(parents=True, exist_ok=True)
    run, prev = runs[-1], (runs[-2] if len(runs) > 1 else None)
    paths = []
    for kind in CHARTS:
        for theme in ("light", "dark"):
            p = folder / f"speed-{kind}-{theme}.svg"
            p.write_text(chart(kind, run, prev, theme, 880, 380 if kind != "prefill" else 340, True) + "\n")
            paths.append(p)
    return paths


def markdown(run: dict) -> str:
    """The decode and prefill tables of one release as Markdown."""
    lines = ["| Context (tokens) | Streams | Tokens/s per stream | Tokens/s in total |", "|---:|---:|---:|---:|"]
    for c in run.get("decode", []):
        lines.append(f"| {c['context']:,} | {c['streams']} | {statistics.mean(c['per_stream']):,.0f} | "
                     f"{statistics.mean(c['total']):,.0f} |")
    lines += ["", "| Prompt (tokens) | Prefill tokens/s | Time to the first token |", "|---:|---:|---:|"]
    for c in run.get("prefill", []):
        v = statistics.mean(c["tokens_per_s"])
        lines.append(f"| {c['context']:,} | {v:,.0f} | {c['context'] / v:.1f} s |")
    return "\n".join(lines)


def cell(run: dict | None, ctx: int, streams: int, key: str) -> float | None:
    """Mean of one decode cell, or None when the release has no such cell."""
    for c in (run or {}).get("decode", []):
        if c["context"] == ctx and c["streams"] == streams:
            return statistics.mean(c[key])
    return None


def prefill_at(run: dict | None, ctx: int) -> float | None:
    for c in (run or {}).get("prefill", []):
        if c["context"] == ctx:
            return statistics.mean(c["tokens_per_s"])
    return None


def change(now: float | None, before: float | None) -> str:
    """A signed percentage with an arrow, or an empty cell."""
    if now is None or before is None:
        return ""
    d = (now / before - 1) * 100
    cls, arrow = ("up", "▲") if d >= 0.5 else (("down", "▼") if d <= -0.5 else ("flat", "●"))
    return f'<span class="{cls}">{arrow} {abs(d):.0f}%</span>'


def slug(run: dict) -> str:
    return "run-" + "".join(ch if ch.isalnum() else "-" for ch in run["release"].lower())


def run_block(run: dict, prev: dict | None, latest: bool) -> str:
    """The section of the page for one release: finding, figures, charts and tables."""
    dec = run.get("decode", [])
    ctxs = sorted({c["context"] for c in dec})
    totals = [(statistics.mean(c["total"]), c) for c in dec]
    best, best_cell = max(totals, key=lambda t: t[0]) if totals else (0, None)
    lo_ctx, hi_ctx = (ctxs[0], ctxs[-1]) if ctxs else (0, 0)
    one_short, one_long = cell(run, lo_ctx, 1, "per_stream"), cell(run, hi_ctx, 1, "per_stream")
    pf_ctx = max((c["context"] for c in run.get("prefill", [])), default=0)
    s = run.get("setup", {})
    vs = f"vs {html.escape(prev['name'])}" if prev else ""

    def tile(label, value, before, unit):
        if value is None:
            return ""
        delta = change(value, before)
        return (f'<div class="tile"><div class="tl">{label}</div><div class="tv">{value:,.0f}<small> {unit}'
                f'</small></div><div class="td">{delta} {vs if delta else ""}</div></div>')

    tiles = [
        tile(f"One stream at {lo_ctx:,} tokens", one_short, cell(prev, lo_ctx, 1, "per_stream"), "tok/s"),
        tile(f"One stream at {hi_ctx:,} tokens", one_long, cell(prev, hi_ctx, 1, "per_stream"), "tok/s"),
        tile(f"Most in total: {stream_name(best_cell['streams'])} at {best_cell['context']:,}" if best_cell else "",
             best if best_cell else None,
             cell(prev, best_cell["context"], best_cell["streams"], "total") if best_cell else None, "tok/s"),
        tile(f"Prefill of one {pf_ctx:,}-token prompt", prefill_at(run, pf_ctx), prefill_at(prev, pf_ctx),
             "tok/s"),
    ]
    rows = []
    for c in dec:
        t, p = statistics.mean(c["total"]), statistics.mean(c["per_stream"])
        rows.append(f'<tr><td>{c["context"]:,}</td><td>{c["streams"]}</td><td>{p:,.0f}</td><td>{t:,.0f}</td>'
                    + (f'<td>{change(t, cell(prev, c["context"], c["streams"], "total"))}</td>' if prev else "")
                    + f'<td>{statistics.mean(c.get("ttft_cached") or [0]):.2f} s</td></tr>')
    prow = [f'<tr><td>{c["context"]:,}</td><td>{statistics.mean(c["tokens_per_s"]):,.0f}</td>'
            + (f'<td>{change(statistics.mean(c["tokens_per_s"]), prefill_at(prev, c["context"]))}</td>'
               if prev else "") + f'<td>{c["context"] / statistics.mean(c["tokens_per_s"]):.1f} s</td></tr>'
            for c in run.get("prefill", [])]

    def fig(kind, wide):
        return (f'<div class="fig"><div class="w">{chart(kind, run, prev, "css", wide, 380, False)}</div>'
                f'<div class="n">{chart(kind, run, prev, "css", 380, 330, False)}</div></div>')

    title = (f"{html.escape(run['name'])} makes {one_short:,.0f} tokens/s for one stream and up to {best:,.0f} "
             f"in total" if one_short else html.escape(run["name"]))
    note = f" ({html.escape(run['note'])})" if run.get("note") else ""
    chg = "<th>Change</th>" if prev else ""
    pchg = "<th>Change</th>" if prev else ""
    return f"""<div class="run" data-id="{slug(run)}"{'' if latest else ' hidden'}>
<section class="band head"><h1>{title}</h1>
<p class="setup">Release {html.escape(run['release'])}{note}, measured {html.escape(run['date'][:10])} on
{html.escape(s.get('gpus', ''))} with tensor parallel {s.get('tensor_parallel', '')}:
{html.escape(s.get('model', ''))}, drafter {html.escape(s.get('drafter', 'none'))}, {html.escape(s.get('kv_cache', ''))}
KV cache, thinking {'on' if s.get('thinking') else 'off'}, {s.get('output_tokens', '')} output tokens per request.
{f'Lighter lines in the charts are {html.escape(prev["name"])}.' if prev else ''}</p>
<div class="tiles">{''.join(tiles)}</div></section>
<section class="band charts"><h2>Total decode speed</h2><p class="sub">Output tokens per second of all streams together.
The bar at a point is the range of the runs.</p>{fig('total', 880)}
<div class="two"><div><h2>Per stream</h2><p class="sub">Output tokens per second that each request receives.</p>
{fig('stream', 560)}</div><div><h2>Prefill</h2><p class="sub">Prompt tokens per second, one prompt that is not in the
cache.</p>{fig('prefill', 560)}</div></div></section>
<section class="band tables"><div class="two wide"><div><h2>Decode</h2><div class="scroll"><table><thead><tr>
<th>Context</th><th>Streams</th><th>Per stream</th><th>Total</th>{chg}<th>First token (cached)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></div><div><h2>Prefill</h2><div class="scroll"><table><thead><tr>
<th>Context</th><th>Tokens/s</th>{pchg}<th>Time</th></tr></thead><tbody>{''.join(prow)}</tbody></table></div></div>
</div></section></div>"""


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>seventy speed sheet</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Space+Grotesk:wght@600&display=swap" rel="stylesheet">
<style>
/* VOICE work: an engine test-bench report, exact and quiet
   IDEA a dyno sheet: each release is a run drawn over the run before it
   PICTURE the total-speed curves across the whole 256K context window, unmeasured lengths shaded
   COLOUR white ground, black ink, cobalt strip, three validated series colours; night #0a0d18
   TYPE Space Grotesk 600 for titles, Inter for text and figures
   LAYOUT cobalt strip with the run picker, finding and four figures, main chart full width, two charts, tables */
:root{color-scheme:light dark;--bg:light-dark(#ffffff,#0a0d18);--ink:light-dark(#0b0b0b,#f0efec);--ink2:light-dark(#52514e,#c3c2b7);--strip:light-dark(#1240e0,#1b3fc4);--pale:light-dark(#eef2ff,#11162a);--rule:light-dark(#e1e0d9,#2a2f3d);--up:light-dark(#006300,#0ca30c);--down:light-dark(#b42323,#e66767);
--c-surface:var(--bg);--c-ink:var(--ink);--c-ink2:var(--ink2);--c-muted:light-dark(#6f6e69,#9a998f);--c-grid:light-dark(#e6e6e3,#1d2130);--c-axis:light-dark(#c3c2b7,#3a3f4e);--c-zone:light-dark(#f2f4fa,#121729);--c-edge:transparent;
--c-s1:light-dark(#2a78d6,#3987e5);--c-s2:light-dark(#eb6834,#d95926);--c-s3:light-dark(#1baf7a,#199e70);--c-s4:light-dark(#eda100,#c98500);--c-s5:light-dark(#e87ba4,#d55181);--c-s6:light-dark(#008300,#0ca30c);--c-s7:light-dark(#6250d6,#9085e9);--c-s8:light-dark(#e34948,#e66767)}
:root[data-theme=light]{color-scheme:light}
:root[data-theme=dark]{color-scheme:dark}
*{box-sizing:border-box;min-width:0}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Inter,system-ui,sans-serif;overflow-x:clip}
h1,h2{font-family:"Space Grotesk",Inter,system-ui,sans-serif;font-weight:600;letter-spacing:-.02em;margin:0}
h1{font-size:clamp(28px,4vw,42px);line-height:1.08;text-wrap:balance;max-width:24em}
h2{font-size:20px;line-height:1.2}
.band{padding:clamp(20px,3vw,40px) max(5vw,16px)}
.strip{background:var(--strip);color:#fff;padding:14px max(5vw,16px);display:flex;flex-wrap:wrap;gap:12px 24px;align-items:center;justify-content:space-between}
.strip b{font-family:"Space Grotesk",Inter,sans-serif;font-size:20px;letter-spacing:-.01em}
.strip{position:relative;z-index:2}
@media(min-width:900px){.strip{position:sticky;top:0}}
.pick{display:flex;flex-wrap:wrap;gap:6px}
.pick button{font:600 14px Inter,sans-serif;color:#fff;background:transparent;border:1px solid rgba(255,255,255,.55);border-radius:8px;min-height:44px;padding:0 14px;cursor:pointer}
.pick button[aria-pressed=true]{background:#fff;color:#1240e0;border-color:#fff}
.pick button:focus-visible{outline:3px solid #fff;outline-offset:2px}
.head{padding-bottom:8px}
.setup{color:var(--ink2);max-width:62em;margin:12px 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,13em),1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
.tile{background:var(--bg);padding:14px 16px}
.tl{color:var(--ink2);font-size:14px}
.tv{font-size:clamp(28px,3.4vw,36px);font-weight:600;line-height:1.2;white-space:nowrap}
.tv small{font-size:14px;font-weight:400;color:var(--ink2)}
.td{font-size:14px;color:var(--ink2);min-height:1.5em}
.up{color:var(--up);font-weight:600;white-space:nowrap}.down{color:var(--down);font-weight:600;white-space:nowrap}.flat{color:var(--ink2);white-space:nowrap}
.sub{color:var(--ink2);margin:4px 0 10px;font-size:15px}
.charts h2{margin-top:8px}
.fig{max-width:880px}
.fig svg{display:block;width:100%;height:auto}
.two .fig{max-width:560px}
.fig .n{display:none}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:24px 40px;margin-top:28px}
.tables{background:var(--pale)}
.tables .two{margin-top:0}
@media(min-width:900px){.two.wide{grid-template-columns:minmax(0,3fr) minmax(0,2fr)}}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;background:var(--bg);border-radius:8px}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-weight:600;font-size:14px;color:var(--ink2)}
th:first-child,td:first-child{text-align:left}
.scroll{overflow-x:auto;margin-top:10px}
.how{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,22em),1fr));gap:8px 40px}
.how p,.how li{color:var(--ink2);max-width:40em}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--pale);padding:12px 14px;border-radius:8px;font-size:14px}
code{overflow-wrap:anywhere}
#tip{position:fixed;z-index:9;pointer-events:none;background:var(--ink);color:var(--bg);font-size:14px;line-height:1.45;padding:8px 10px;border-radius:8px;max-width:280px}
[data-tip]{cursor:default;outline:none}
[data-tip]:focus-visible{stroke:var(--ink);stroke-width:2}
@media(max-width:600px){.fig .w{display:none}.fig .n{display:block}.fig,.two .fig{max-width:100%}table{font-size:13px}th,td{padding:6px 6px}}
</style></head><body>
<header class="strip"><b>seventy speed sheet</b><nav class="pick" aria-label="Release">__PICK__</nav></header>
<main>__RUNS__
<section class="band how"><div><h2>How the speed is measured</h2>
<p>Each decode test sends requests whose prompt is already in the prefix cache, so the test measures only the
generation of tokens. Per stream is the output rate of one request from its first to its last token. Total is the output
rate of all requests together while all of them run (results made before bench/sweep.py add up the per-stream rates
instead, which gives more when the streams start far apart). When a cell runs more than one time, the bar at a point shows the range of the
runs.</p>
<p>The prefill test sends one prompt that is not in the cache and measures the time to its first token.</p></div>
<div><h2>Add a release</h2><p>Measure a running server into a results file in <code>bench/results/</code>, then make
the charts and this page again:</p><pre><code>python bench/sweep.py --release 1.1.0 --out bench/results/1.1.0.json
python bench/report.py --images docs/images --html speed-sheet.html</code></pre>
<p>Each release is drawn over the release before it. The format is in the docstring of <code>bench/report.py</code>.</p></div></section>
</main><div id="tip" hidden></div>
<script>
{
const runs=[...document.querySelectorAll('.run')];
const btns=[...document.querySelectorAll('.pick button')];
const show=id=>{const r=runs.find(x=>x.dataset.id===id)||runs[0];runs.forEach(x=>x.hidden=x!==r);btns.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.run===r.dataset.id)))};
btns.forEach(b=>b.addEventListener('click',()=>{location.hash=b.dataset.run;show(b.dataset.run)}));
window.addEventListener('hashchange',()=>show(location.hash.slice(1)));
show(location.hash.slice(1)||runs[0].dataset.id);
const tip=document.getElementById('tip');
const fill=t=>{tip.replaceChildren(...t.dataset.tip.split('|').map((l,i)=>{const e=document.createElement(i?'div':'b');e.textContent=l;return e}));tip.hidden=false};
const place=(x,y)=>{tip.style.left=Math.max(8,Math.min(x+14,innerWidth-tip.offsetWidth-8))+'px';tip.style.top=Math.min(y+14,innerHeight-tip.offsetHeight-8)+'px'};
document.addEventListener('pointerover',e=>{const t=e.target.closest('[data-tip]');if(t){fill(t);place(e.clientX,e.clientY)}else tip.hidden=true});
document.addEventListener('pointermove',e=>{if(!tip.hidden)place(e.clientX,e.clientY)});
document.addEventListener('focusin',e=>{const t=e.target.closest('[data-tip]');if(t){fill(t);const r=t.getBoundingClientRect();place(r.right,r.top)}else tip.hidden=true});
}
</script></body></html>
"""


def write_html(runs: list[dict], path: Path) -> None:
    """One page with every release, newest first in the picker; each release is drawn over the one before it."""
    blocks, pick = [], []
    for i in range(len(runs) - 1, -1, -1):
        run, prev = runs[i], (runs[i - 1] if i else None)
        blocks.append(run_block(run, prev, i == len(runs) - 1))
        pick.append(f'<button type="button" data-run="{slug(run)}" aria-pressed="false">'
                    f'{html.escape(run["name"])}</button>')
    path.write_text(PAGE.replace("__PICK__", "".join(pick)).replace("__RUNS__", "\n".join(blocks)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--images", type=Path, help="folder for the README charts")
    ap.add_argument("--html", type=Path, help="file for the speed sheet")
    ap.add_argument("--markdown", action="store_true", help="print the tables of the newest release")
    args = ap.parse_args()
    runs = load(args.results)
    if not runs:
        raise SystemExit(f"no results files in {args.results}")
    if args.images:
        for p in write_images(runs, args.images):
            print(p)
    if args.markdown:
        print(markdown(runs[-1]))
    if args.html:
        write_html(runs, args.html)
        print(args.html)


if __name__ == "__main__":
    main()
