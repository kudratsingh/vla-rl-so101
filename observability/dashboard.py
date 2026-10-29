"""Make an offline SO-101 research dashboard from existing local run records.

No imports from the training, simulator, device, cloud, or servo code. No network I/O.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeGuard

RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
METRIC_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,95}\Z")
SENSITIVE = re.compile(
    r"password|secret|api.?key|access.?token|authorization|credential", re.IGNORECASE
)
BENCHMARK_TASKS = ("PickPlace", "Push", "Stack")
BENCHMARK_CELLS = ("seen", "shifted", "unseen_objects")
METRIC_PREFIXES = ("train/", "rollout/", "eval/", "time/")
METRIC_FIELDS = {
    "loss",
    "l1_norm",
    "lr",
    "grad_norm",
    "s_per_step",
    "samples_per_s",
    "max_mem_gb",
    "step_s",
    "pg_loss",
    "clip_frac",
    "approx_kl",
    "entropy",
    "gradient_norm",
}
MAX_RUNS = 50
MAX_METRICS = 120
MAX_SAMPLES = 2400
MAX_LOG_BYTES = 32 * 1024 * 1024


def _numeric(value: object) -> TypeGuard[int | float]:
    return (type(value) is int or type(value) is float) and math.isfinite(value)


def _valid_metric(key: object) -> bool:
    return (
        isinstance(key, str)
        and METRIC_NAME.fullmatch(key) is not None
        and not SENSITIVE.search(key)
        and (key in METRIC_FIELDS or key.startswith(METRIC_PREFIXES))
    )


def _read_events(
    path: Path, series: dict[str, list[list[float]]]
) -> dict[tuple[str, str, int], tuple[int, int]]:
    """Stream JSONL; only extract numeric scalar telemetry and sim eval counts.

    Evaluation entries are keyed by task, cell and seed: repeated rows replace rather than
    silently double count. Event rows may still be in progress when inspected.
    """
    evals: dict[tuple[str, str, int], tuple[int, int]] = {}
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_LOG_BYTES:
        return evals
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue  # E.g. writer has not flushed the last line.
            if not isinstance(event, dict):
                continue
            step = event.get("step")
            if type(step) is int and step >= 0:
                for key, value in event.items():
                    if not _valid_metric(key) or not _numeric(value):
                        continue
                    if key not in series and len(series) >= MAX_METRICS:
                        continue
                    samples = series.setdefault(key, [])
                    samples.append([float(step), float(value)])
                    if len(samples) > MAX_SAMPLES:
                        samples[:] = samples[::2] + ([samples[-1]] if len(samples) % 2 == 0 else [])
            task, cell, seed = event.get("task"), event.get("cell"), event.get("seed")
            successes, episodes = event.get("successes"), event.get("episodes")
            if (
                task in BENCHMARK_TASKS
                and cell in BENCHMARK_CELLS
                and type(seed) is int
                and seed >= 0
                and type(successes) is int
                and type(episodes) is int
                and 0 <= successes <= episodes
                and 0 < episodes <= 100_000
            ):
                evals[(task, cell, seed)] = (successes, episodes)
    return evals


def _wilson(successes: int, episodes: int) -> list[float]:
    """95% Wilson score interval, expressed as proportions."""
    p = successes / episodes
    z = 1.959963984540054
    z2 = z * z
    denom = 1 + z2 / episodes
    center = (p + z2 / (2 * episodes)) / denom
    half = z * math.sqrt(p * (1 - p) / episodes + z2 / (4 * episodes**2)) / denom
    return [
        0.0 if successes == 0 else max(0.0, center - half),
        1.0 if successes == episodes else min(1.0, center + half),
    ]


def _benchmarks(evals: dict[tuple[str, str, int], tuple[int, int]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for (task, cell, _seed), pair in evals.items():
        by_cell[(task, cell)].append(pair)
    result = []
    for task in BENCHMARK_TASKS:
        for cell in BENCHMARK_CELLS:
            results = by_cell.get((task, cell))
            if results:
                successes = sum(p[0] for p in results)
                episodes = sum(p[1] for p in results)
                result.append(
                    {
                        "task": task,
                        "cell": cell,
                        "successes": successes,
                        "episodes": episodes,
                        "seeds": len(results),
                        "rate": successes / episodes,
                        "ci": _wilson(successes, episodes),
                    }
                )
    return result


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Read only regular files in immediate run folders; never follow per-run symlinks."""
    if not root.is_dir():
        return []
    collected: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir(), key=lambda x: x.name, reverse=True):
        if len(collected) >= MAX_RUNS:
            break
        if not folder.is_dir() or folder.is_symlink() or not RUN_NAME.fullmatch(folder.name):
            continue
        record = folder / "record.json"
        if not record.is_file() or record.is_symlink():
            continue
        try:
            metadata = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        metric_series: dict[str, list[list[float]]] = {}
        logs = folder / "metrics.jsonl"
        # The frozen sim evaluation protocol logs rows through scripts/wo02_eval.py.
        # Never mix future real-hardware data into this simulation comparison.
        is_sim_eval = metadata.get("script") == "scripts/wo02_eval.py"
        evals = _read_events(logs, metric_series)
        results = metadata.get("results")
        protocol_hash = results.get("protocol_hash") if isinstance(results, dict) else None
        if not (
            is_sim_eval
            and isinstance(protocol_hash, str)
            and re.fullmatch(r"[0-9a-f]{64}", protocol_hash)
        ):
            protocol_hash = None
        status = metadata.get("status")
        cost = metadata.get("actual_usd")
        git = metadata.get("git")
        commit = git.get("commit") if isinstance(git, dict) else None
        collected.append(
            {
                "id": folder.name,
                "status": status if status in ("running", "ok", "failed") else "unknown",
                "commit": (
                    commit[:8]
                    if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit)
                    else None
                ),
                "cost_usd": float(cost) if _numeric(cost) and cost >= 0 else None,
                "series": metric_series,
                "protocol_hash": protocol_hash,
                "benchmarks": _benchmarks(evals) if is_sim_eval else [],
            }
        )
    return collected


# A single self-contained HTML page: no CDN, WebSockets, client network requests or login.
HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO-101 · Research Observatory</title><style>
:root{color-scheme:dark;background:#0d1321;color:#e6eaf2;font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{max-width:1250px;margin:0 auto;padding:32px 28px 72px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap;margin-bottom:26px}
h1{font-size:clamp(27px,3vw,40px);letter-spacing:-.04em;margin:6px 0}
h2{font-size:18px;margin:0 0 6px;letter-spacing:-.02em}
.eyebrow{color:#9fc8f6;text-transform:uppercase;letter-spacing:.15em;font-size:11px;font-weight:700}
p,small{color:#a7b4c7;line-height:1.45;margin:0}small{display:block;margin-top:8px}
.panel{background:#141e30;border:1px solid #2b3850;border-radius:16px;padding:22px;margin:18px 0}
.row{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.tag{border:1px solid #3c4b61;border-radius:999px;padding:8px 12px;color:#c1cbdc;font-size:12px}
select{background:#1a2940;color:#e6eaf2;border:1px solid #516179;border-radius:9px;padding:10px 12px;max-width:100%;min-width:170px}
select:focus-visible,input:focus-visible{outline:2px solid #8ec6ff}
#chart{display:block;width:100%;height:auto;min-height:240px;margin-top:12px}
.chart-label{font-size:12px;fill:#a0b0c8}.gridline{stroke:#34435b;stroke-width:1}
#legend{display:flex;flex-wrap:wrap;gap:8px 18px}.legend{display:flex;align-items:center;gap:7px;font-size:12px;color:#c3cfe1}
.swatch{width:12px;height:12px;display:inline-block;border-radius:3px}
.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}
th,td{text-align:left;padding:12px;border-bottom:1px solid #2b3850}th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:#9cabca}td{color:#e2e8f4}
#benchmarkBody tr:last-child td,#runBody tr:last-child td{border-bottom:0}
.empty{color:#a9b9cd;padding:30px 12px;text-align:center}.pos{color:#6bd9a4}.neg{color:#f1a3a3}
footer{color:#8495ae;font-size:12px;margin-top:22px}
</style></head><body>
<header><div><div class="eyebrow">Local-first · Simulation research</div><h1>SO-101 Observatory</h1><p>Training signal, seed-aware outcomes and run provenance.</p></div><div class="tag">Offline · Read-only inputs</div></header>
<section class="panel"><div class="row"><div><h2>SFT / GRPO metric explorer</h2><p>Compare runs on like-for-like metrics, including rollout group drops and held-out success.</p></div><label for="metric">Metric <select id="metric"></select></label></div><svg id="chart" role="img" aria-label="Selected metric over training steps" viewBox="0 0 960 345"></svg><div id="legend"></div></section>
<section class="panel"><h2>Simulation evaluation · two-run comparison</h2><p>Explicitly choose a baseline and candidate. Selection does not imply model type; verify checkpoint provenance. Real trials are excluded.</p><div class="row" style="margin:16px 0"><label for="baseline">Baseline <select id="baseline"></select></label><label for="candidate">Candidate <select id="candidate"></select></label></div><p id="protocolStatus" role="status" style="overflow-wrap:anywhere"></p><div class="tablewrap"><table><thead><tr><th>Task</th><th>Distribution</th><th>Baseline (95% CI)</th><th>Candidate (95% CI)</th><th>Δ pp</th><th>Trials</th></tr></thead><tbody id="benchmarkBody"></tbody></table></div><small>Each cell aggregates distinct recorded evaluation seeds; CIs use the 95% Wilson interval. Differences are descriptive, not significance tests. Missing evaluation data is not zero success.</small></section>
<section class="panel"><h2>Run inventory</h2><p>Recorded local run status and estimated cost, not live RunPod billing.</p><div class="tablewrap"><table><thead><tr><th>Run</th><th>Status</th><th>Steps seen</th><th>Latest success (train tasks)</th><th>Recorded cost</th><th>Commit</th></tr></thead><tbody id="runBody"></tbody></table></div></section>
<footer>Uses only numerical metrics and (for sim eval only) task/cell/seed aggregate counts from record.json and metrics.jsonl. No frames, prompts, trajectories, checkpoints, robot commands, secrets, full configuration or live hardware endpoints are included.</footer>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const runs=JSON.parse(document.getElementById("data").textContent),ns="http://www.w3.org/2000/svg";
const colors=["#60a5fa","#52d4b6","#f4c16e","#e88fb4","#a78bfa","#4ecdec","#f59673"];
const metric=document.getElementById("metric"),chart=document.getElementById("chart"),legend=document.getElementById("legend");
const enabled=new Set(runs.map(r=>r.id));
function node(tag,attrs,text){const x=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs||{}))x.setAttribute(k,String(v));if(text!==undefined)x.textContent=text;chart.appendChild(x);return x;}
function cell(tr,value,cls){const td=document.createElement("td");td.textContent=value;if(cls)td.className=cls;tr.appendChild(td);}
function fmt(n){return n===null||!Number.isFinite(n)?"—":Number(n).toLocaleString(undefined,{maximumFractionDigits:4});}
function latest(r,key){const p=r.series[key]||[];return p.length?p[p.length-1][1]:null;}
const keys=[...new Set(runs.flatMap(r=>Object.keys(r.series)))];
const favorites=["eval/success_train","eval/success","rollout/success_rate","rollout/frac_groups_dropped","rollout/group_reward_std","train/pg_loss","train/approx_kl","train/clip_frac","train/entropy","loss","grad_norm"];
keys.sort((a,b)=>(favorites.indexOf(a)<0?99:favorites.indexOf(a))-(favorites.indexOf(b)<0?99:favorites.indexOf(b))||a.localeCompare(b));
for(const key of keys){const o=document.createElement("option");o.value=key;o.textContent=key;metric.appendChild(o);}
function renderChart(){chart.replaceChildren();legend.replaceChildren();const key=metric.value;
 const active=runs.map((r,i)=>({r,i,p:r.series[key]||[]})).filter(x=>enabled.has(x.r.id)&&x.p.length);
 if(!active.length){node("text",{x:480,y:165,"text-anchor":"middle",class:"chart-label"},key?"No enabled runs have this metric":"No numeric training metrics found");}
 else {let xmin=Number.POSITIVE_INFINITY,xmax=Number.NEGATIVE_INFINITY,ymin=Number.POSITIVE_INFINITY,ymax=Number.NEGATIVE_INFINITY;
 for(const {p} of active)for(const [x,y] of p){xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
 if(xmin===xmax)xmax=xmin+1;if(ymin===ymax){const pad=Math.abs(ymin)*.05||.1;ymin-=pad;ymax+=pad;}
 const left=80,right=934,top=23,bottom=300,px=x=>left+(x-xmin)*(right-left)/(xmax-xmin),py=y=>bottom-(y-ymin)*(bottom-top)/(ymax-ymin);
 for(let t=0;t<=4;t++){const v=ymin+t*(ymax-ymin)/4,y=py(v);node("line",{x1:left,y1:y,x2:right,y2:y,class:"gridline"});node("text",{x:left-9,y:y+4,"text-anchor":"end",class:"chart-label"},fmt(v));const x=xmin+t*(xmax-xmin)/4;node("text",{x:px(x),y:326,"text-anchor":"middle",class:"chart-label"},fmt(x));}
 node("text",{x:505,y:343,"text-anchor":"middle",class:"chart-label"},"Training step");
 for(const {i,p} of active)node("polyline",{points:p.map(([x,y])=>px(x).toFixed(2)+","+py(y).toFixed(2)).join(" "),fill:"none",stroke:colors[i%colors.length],"stroke-width":2.4,"stroke-linejoin":"round"});
 }
 runs.forEach((r,i)=>{const label=document.createElement("label");label.className="legend";const cb=document.createElement("input");cb.type="checkbox";cb.checked=enabled.has(r.id);cb.addEventListener("change",()=>{if(cb.checked)enabled.add(r.id);else enabled.delete(r.id);renderChart();});const swatch=document.createElement("span");swatch.className="swatch";swatch.style.background=colors[i%colors.length];const name=document.createElement("span");name.textContent=r.id;label.append(cb,swatch,name);legend.appendChild(label);});
}
metric.addEventListener("change",renderChart);renderChart();
for(const r of runs){const tr=document.createElement("tr");let maxStep=null;for(const series of Object.values(r.series))for(const point of series)maxStep=maxStep===null?point[0]:Math.max(maxStep,point[0]);cell(tr,r.id);cell(tr,r.status);cell(tr,fmt(maxStep));const latestSuccess=latest(r,"eval/success_train");cell(tr,latestSuccess===null?"—":(latestSuccess*100).toFixed(1)+"%");cell(tr,r.cost_usd===null?"—":"$"+fmt(r.cost_usd));cell(tr,r.commit||"—");document.getElementById("runBody").appendChild(tr);}
if(!runs.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=6;td.className="empty";td.textContent="No run records found. Choose a local runs/ directory.";tr.appendChild(td);document.getElementById("runBody").appendChild(tr);}
const evalRuns=runs.filter(r=>r.benchmarks.length),baseBox=document.getElementById("baseline"),candBox=document.getElementById("candidate");
for(const sel of [baseBox,candBox]){const empty=document.createElement("option");empty.value="";empty.textContent="Not selected";sel.appendChild(empty);for(const r of evalRuns){const o=document.createElement("option");o.value=r.id;o.textContent=r.id;sel.appendChild(o);}}
if(evalRuns.length){baseBox.value=evalRuns[evalRuns.length-1].id;if(evalRuns.length>1)candBox.value=evalRuns[0].id;}
const tasks=["PickPlace","Push","Stack"],cells=["seen","shifted","unseen_objects"];
function bcell(r,t,c){return r?.benchmarks.find(x=>x.task===t&&x.cell===c)||null;}
function score(b,r){if(!b)return "—";if(!r.protocol_hash)return "Unverified protocol";return (b.rate*100).toFixed(1)+"% ["+(b.ci[0]*100).toFixed(1)+", "+(b.ci[1]*100).toFixed(1)+"]";}
function renderBench(){const tbody=document.getElementById("benchmarkBody");tbody.replaceChildren();const a=evalRuns.find(r=>r.id===baseBox.value),b=evalRuns.find(r=>r.id===candBox.value);
const comparable=Boolean(a?.protocol_hash&&b?.protocol_hash&&a.protocol_hash===b.protocol_hash);
const provenance=[];
for(const [label,r] of [["Baseline",a],["Candidate",b]])if(r)provenance.push(label+" "+r.id+" protocol: "+(r.protocol_hash||"missing or invalid; success rates hidden"));
if(a&&b&&!comparable)provenance.push(a.protocol_hash&&b.protocol_hash?"Protocol mismatch; delta withheld.":"Matching recorded protocol hashes required; delta withheld.");
document.getElementById("protocolStatus").textContent=provenance.join(" · ");
if(!a&&!b){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=6;td.className="empty";td.textContent="No local simulation evaluation rows. Copy metrics.jsonl from a WO-02 evaluation run.";tr.appendChild(td);tbody.appendChild(tr);return;}
for(const t of tasks)for(const c of cells){const x=bcell(a,t,c),y=bcell(b,t,c),tr=document.createElement("tr");cell(tr,t);cell(tr,c.replaceAll("_"," "));cell(tr,score(x,a));cell(tr,score(y,b));const diff=comparable&&x&&y?(y.rate-x.rate)*100:null;cell(tr,diff===null?"—":(diff>=0?"+":"")+diff.toFixed(1),diff===null?"":diff>0?"pos":diff<0?"neg":"");cell(tr,[x?`${x.episodes} / ${x.seeds} seeds`:null,y?`${y.episodes} / ${y.seeds} seeds`:null].filter(Boolean).join(" | ")||"—");tbody.appendChild(tr);}}
baseBox.addEventListener("change",renderBench);candBox.addEventListener("change",renderBench);renderBench();
</script></body></html>"""


def write_dashboard(runs_root: Path, output: Path) -> int:
    """Write only an offline report. Never overwrite a run record or training artifact."""
    root = runs_root.resolve()
    destination = output.resolve()
    if destination.is_relative_to(root) and (
        not destination.relative_to(root).parts
        or destination.relative_to(root).parts[0] != "observability"
    ):
        raise ValueError("output inside runs/ must be in runs/observability/")
    data = load_runs(root)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", dir=destination.parent, encoding="utf-8", delete=False
    ) as file:
        temp = Path(file.name)
        try:
            file.write(HTML.replace("__PAYLOAD__", payload))
            file.flush()
            os.fsync(file.fileno())
            os.chmod(temp, 0o600)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, destination)
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, help="default: <runs-root>/observability/index.html")
    args = parser.parse_args()
    output = args.output or args.runs_root / "observability" / "index.html"
    try:
        count = write_dashboard(args.runs_root, output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Read {count} local runs; dashboard: {output}")


if __name__ == "__main__":
    main()
