import argparse
import html
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "saved_models" / "dashboard_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATE = {
    "process": None,
    "mode": "idle",
    "log_path": None,
    "command": "",
    "started_at": None,
    "returncode": None,
}
LOCK = threading.Lock()

DEFAULTS = {
    "anchor_path": r".\emo_anchors\sup-simcse-roberta-large",
    "bert_path": r".\pretrained\sup-simcse-roberta-large",
    "dataset_name": "IEMOCAP",
    "gpu_id": "1",
    "seed": "1",
    "epochs": "8",
    "batch_size": "8",
    "lr": "0.0004",
    "ptmlr": "1e-05",
    "dropout": "0.1",
    "temp": "0.1",
    "ce_loss_weight": "0.1",
    "angle_loss_weight": "0.1",
    "stage_two_lr": "1e-4",
    "num_subanchors": "4",
    "prototype_pooling": "domain_gated",
    "domain_entropy_eps": "1e-6",
    "prototype_momentum": "0.9",
    "extra_args": "--disable_training_progress_bar --use_nearest_neighbour --disable_two_stage_training",
}

STYLE = r"""
:root {
  --ink: #17201a;
  --muted: #66756c;
  --line: rgba(23, 32, 26, .12);
  --panel: rgba(255, 252, 244, .84);
  --panel-strong: rgba(255, 252, 244, .96);
  --accent: #f06a3a;
  --accent-2: #1b7f6b;
  --accent-3: #f3b94d;
  --shadow: 0 24px 80px rgba(41, 54, 45, .22);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  color: var(--ink);
  font-family: "Segoe UI", "Microsoft YaHei", "Noto Sans SC", sans-serif;
  background:
    radial-gradient(circle at 18% 12%, rgba(240, 106, 58, .28), transparent 26rem),
    radial-gradient(circle at 82% 8%, rgba(27, 127, 107, .28), transparent 30rem),
    linear-gradient(135deg, #fff4d6 0%, #f3f7e9 48%, #e8f4f0 100%);
}
body:before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image: linear-gradient(rgba(23,32,26,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(23,32,26,.045) 1px, transparent 1px);
  background-size: 38px 38px;
  mask-image: linear-gradient(to bottom, black, transparent 82%);
}
.wrap { position: relative; max-width: 1280px; margin: 0 auto; padding: 34px 24px 42px; }
.hero { display: grid; grid-template-columns: 1.2fr .8fr; gap: 18px; align-items: stretch; margin-bottom: 18px; }
.card { background: var(--panel); backdrop-filter: blur(18px); border: 1px solid var(--line); box-shadow: var(--shadow); border-radius: 28px; }
.title { padding: 30px; overflow: hidden; position: relative; }
.title h1 { margin: 0; font-size: clamp(32px, 5vw, 64px); line-height: .95; letter-spacing: -.055em; }
.title p { margin: 18px 0 0; color: var(--muted); max-width: 760px; font-size: 16px; line-height: 1.7; }
.badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
.badge { padding: 8px 11px; border-radius: 999px; background: rgba(255,255,255,.62); border: 1px solid var(--line); font-size: 13px; color: #344139; }
.status { padding: 26px; display: flex; flex-direction: column; justify-content: space-between; gap: 18px; }
.status-pill { display: inline-flex; align-items: center; gap: 10px; width: fit-content; padding: 12px 15px; border-radius: 999px; background: #17201a; color: #fff9ea; font-weight: 700; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent-3); box-shadow: 0 0 0 8px rgba(243,185,77,.2); }
.status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.metric { padding: 14px; border-radius: 18px; background: rgba(255,255,255,.58); border: 1px solid var(--line); }
.metric b { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
.metric span { font-weight: 800; word-break: break-all; }
.main { display: grid; grid-template-columns: 430px 1fr; gap: 18px; }
.form { padding: 22px; }
.form h2, .console h2 { margin: 0 0 16px; font-size: 18px; letter-spacing: -.02em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.field { display: flex; flex-direction: column; gap: 6px; }
.field.wide { grid-column: 1 / -1; }
label { font-size: 12px; color: var(--muted); font-weight: 700; }
input, select, textarea {
  width: 100%; border: 1px solid var(--line); border-radius: 14px; padding: 11px 12px;
  background: rgba(255,255,255,.74); color: var(--ink); outline: none; font: inherit; font-size: 13px;
}
textarea { resize: vertical; min-height: 72px; }
input:focus, select:focus, textarea:focus { border-color: rgba(27,127,107,.65); box-shadow: 0 0 0 4px rgba(27,127,107,.12); }
.actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }
button {
  border: 0; border-radius: 16px; padding: 13px 14px; font-weight: 800; cursor: pointer; color: #fffaf0;
  background: linear-gradient(135deg, var(--accent), #d84629); box-shadow: 0 12px 28px rgba(240,106,58,.24);
}
button.secondary { background: linear-gradient(135deg, var(--accent-2), #145f52); box-shadow: 0 12px 28px rgba(27,127,107,.2); }
button.ghost { color: var(--ink); background: rgba(255,255,255,.74); border: 1px solid var(--line); box-shadow: none; }
.console { padding: 22px; min-width: 0; }
.toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.cmd { padding: 12px 14px; border-radius: 16px; background: rgba(23,32,26,.92); color: #fff3d6; font-family: Consolas, monospace; font-size: 12px; overflow: auto; white-space: nowrap; }
pre {
  margin: 12px 0 0; height: 590px; overflow: auto; padding: 18px; border-radius: 22px;
  background: #101712; color: #d7f8df; border: 1px solid rgba(255,255,255,.1);
  font-family: Consolas, "Cascadia Mono", monospace; font-size: 13px; line-height: 1.55;
}
.toast { margin-top: 12px; color: var(--muted); font-size: 13px; min-height: 18px; }
@media (max-width: 980px) { .hero, .main { grid-template-columns: 1fr; } .grid { grid-template-columns: 1fr; } pre { height: 420px; } }
"""

HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>EACL Training Dashboard</title>
<style>__STYLE__</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <div class="card title">
      <h1>ERC Training<br/>Control Room</h1>
      <p>给你的远程 Windows 服务器准备的轻量训练面板。它不会开桌面弹窗，而是在 VS Code 端口转发里打开浏览器界面，更稳，也更不容易被远程会话坑到。</p>
      <div class="badges">
        <span class="badge">4-domain anchors</span><span class="badge">domain_gated</span><span class="badge">live log</span><span class="badge">no extra deps</span>
      </div>
    </div>
    <div class="card status">
      <div class="status-pill"><span class="dot"></span><span id="statusText">idle</span></div>
      <div class="status-grid">
        <div class="metric"><b>PID</b><span id="pidText">-</span></div>
        <div class="metric"><b>Return</b><span id="returnText">-</span></div>
        <div class="metric"><b>Started</b><span id="startedText">-</span></div>
        <div class="metric"><b>Log</b><span id="logText">-</span></div>
      </div>
    </div>
  </section>
  <section class="main">
    <div class="card form">
      <h2>训练参数</h2>
      <div class="grid" id="formGrid"></div>
      <div class="actions">
        <button onclick="startTrain()">开始训练</button>
        <button class="secondary" onclick="generateAnchors()">生成 Anchors</button>
        <button class="ghost" onclick="stopTrain()">停止训练</button>
        <button class="ghost" onclick="copyCommand()">复制命令</button>
      </div>
      <div class="toast" id="toast"></div>
    </div>
    <div class="card console">
      <div class="toolbar"><h2>实时日志</h2><button class="ghost" onclick="refreshNow()">刷新</button></div>
      <div class="cmd" id="cmdText">等待命令...</div>
      <pre id="logBox">日志会显示在这里。</pre>
    </div>
  </section>
</div>
<script>
const defaults = __DEFAULTS__;
const fields = [
  ['anchor_path','Anchor 路径','wide'], ['bert_path','预训练模型路径','wide'],
  ['dataset_name','数据集'], ['gpu_id','GPU ID'], ['seed','Seed'], ['epochs','Epochs'],
  ['batch_size','Batch'], ['lr','LR'], ['ptmlr','PLM LR'], ['dropout','Dropout'],
  ['temp','Temp'], ['ce_loss_weight','CE 权重'], ['angle_loss_weight','Angle 权重'], ['stage_two_lr','二阶段 LR'],
  ['num_subanchors','子锚点/域数'], ['prototype_pooling','聚合方式'], ['prototype_momentum','Anchor momentum'], ['domain_entropy_eps','Entropy eps'],
  ['extra_args','额外参数','wide']
];
function esc(s){ return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function buildForm(){
  const grid = document.getElementById('formGrid');
  grid.innerHTML = fields.map(([id,label,wide]) => {
    if(id === 'prototype_pooling') return `<div class="field ${wide||''}"><label>${label}</label><select id="${id}"><option>domain_gated</option><option>entropy</option><option>max</option><option>logsumexp</option></select></div>`;
    if(id === 'dataset_name') return `<div class="field ${wide||''}"><label>${label}</label><select id="${id}"><option>IEMOCAP</option><option>MELD</option><option>EmoryNLP</option></select></div>`;
    if(id === 'extra_args') return `<div class="field wide"><label>${label}</label><textarea id="${id}">${esc(defaults[id])}</textarea></div>`;
    return `<div class="field ${wide||''}"><label>${label}</label><input id="${id}" value="${esc(defaults[id])}" /></div>`;
  }).join('');
  for (const [k,v] of Object.entries(defaults)) { const el = document.getElementById(k); if(el) el.value = v; }
}
function payload(){ const data = {}; for(const [id] of fields){ data[id] = document.getElementById(id).value; } return data; }
function toast(s){ document.getElementById('toast').textContent = s; }
async function post(path){
  const res = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload())});
  const data = await res.json(); if(!res.ok) throw new Error(data.error || 'request failed'); return data;
}
async function startTrain(){ try { const d = await post('/api/start'); toast(d.message); await refreshNow(); } catch(e){ toast(e.message); } }
async function generateAnchors(){ try { const d = await post('/api/generate'); toast(d.message); await refreshNow(); } catch(e){ toast(e.message); } }
async function stopTrain(){ try { const d = await fetch('/api/stop', {method:'POST'}).then(r=>r.json()); toast(d.message); await refreshNow(); } catch(e){ toast(e.message); } }
async function copyCommand(){ await navigator.clipboard.writeText(document.getElementById('cmdText').textContent); toast('命令已复制'); }
async function refreshNow(){
  const s = await fetch('/api/status').then(r=>r.json());
  document.getElementById('statusText').textContent = s.mode;
  document.getElementById('pidText').textContent = s.pid || '-';
  document.getElementById('returnText').textContent = s.returncode ?? '-';
  document.getElementById('startedText').textContent = s.started_at || '-';
  document.getElementById('logText').textContent = s.log_name || '-';
  document.getElementById('cmdText').textContent = s.command || '等待命令...';
  const log = await fetch('/api/log').then(r=>r.text());
  const box = document.getElementById('logBox');
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  box.textContent = log || '日志会显示在这里。';
  if(nearBottom) box.scrollTop = box.scrollHeight;
}
buildForm(); refreshNow(); setInterval(refreshNow, 2500);
</script>
</body>
</html>
""".replace("__STYLE__", STYLE).replace("__DEFAULTS__", json.dumps(DEFAULTS, ensure_ascii=False))


def split_extra_args(value):
    return [part for part in value.split() if part.strip()]


def build_train_command(data):
    cmd = [
        sys.executable, "src/run.py",
        "--anchor_path", data["anchor_path"],
        "--bert_path", data["bert_path"],
        "--dataset_name", data["dataset_name"],
        "--gpu_id", data["gpu_id"],
        "--ce_loss_weight", data["ce_loss_weight"],
        "--temp", data["temp"],
        "--seed", data["seed"],
        "--angle_loss_weight", data["angle_loss_weight"],
        "--stage_two_lr", data["stage_two_lr"],
        "--num_subanchors", data["num_subanchors"],
        "--prototype_pooling", data["prototype_pooling"],
        "--domain_entropy_eps", data["domain_entropy_eps"],
        "--prototype_momentum", data["prototype_momentum"],
        "--dropout", data["dropout"],
        "--lr", data["lr"],
        "--ptmlr", data["ptmlr"],
        "--batch_size", data["batch_size"],
        "--epochs", data["epochs"],
    ]
    cmd.extend(split_extra_args(data.get("extra_args", "")))
    return cmd


def build_anchor_command(data):
    return [sys.executable, "src/generate_anchors.py", "--bert_path", data["bert_path"], "--num_subanchors", data["num_subanchors"]]


def quote_cmd(cmd):
    return subprocess.list2cmdline(cmd)


def launch(cmd, mode):
    with LOCK:
        proc = STATE.get("process")
        if proc is not None and proc.poll() is None:
            raise RuntimeError("已有任务正在运行，请先停止或等待结束。")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"{mode}_{stamp}.log"
        log_file = open(log_path, "w", encoding="utf-8", errors="replace")
        log_file.write(f"# cwd: {ROOT}\n# command: {quote_cmd(cmd)}\n\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        STATE.update({"process": proc, "mode": mode, "log_path": log_path, "command": quote_cmd(cmd), "started_at": datetime.now().strftime("%H:%M:%S"), "returncode": None})
        threading.Thread(target=watch_process, args=(proc, log_file), daemon=True).start()


def watch_process(proc, log_file):
    rc = proc.wait()
    try:
        log_file.write(f"\n# process exited with code {rc}\n")
        log_file.close()
    except Exception:
        pass
    with LOCK:
        if STATE.get("process") is proc:
            STATE["returncode"] = rc
            STATE["mode"] = "done" if rc == 0 else "failed"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_text(self, text, status=200, content_type="text/plain; charset=utf-8"):
        body = text.encode("utf-8", errors="replace")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status=200):
        self.send_text(json.dumps(obj, ensure_ascii=False), status, "application/json; charset=utf-8")

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(raw or "{}")
        merged = DEFAULTS.copy()
        merged.update({k: str(v) for k, v in data.items()})
        return merged

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_text(HTML, content_type="text/html; charset=utf-8")
        elif path == "/api/status":
            with LOCK:
                proc = STATE.get("process")
                log_path = STATE.get("log_path")
                status = {
                    "mode": STATE["mode"],
                    "pid": proc.pid if proc is not None and proc.poll() is None else None,
                    "returncode": STATE.get("returncode") if proc is None or proc.poll() is not None else None,
                    "command": STATE.get("command", ""),
                    "started_at": STATE.get("started_at"),
                    "log_name": log_path.name if log_path else None,
                }
            self.send_json(status)
        elif path == "/api/log":
            with LOCK:
                log_path = STATE.get("log_path")
            if not log_path or not Path(log_path).exists():
                self.send_text("")
            else:
                data = Path(log_path).read_text(encoding="utf-8", errors="replace")
                self.send_text(data[-120000:])
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/start":
                data = self.read_json()
                launch(build_train_command(data), "train")
                self.send_json({"message": "训练已启动。"})
            elif path == "/api/generate":
                data = self.read_json()
                launch(build_anchor_command(data), "anchors")
                self.send_json({"message": "Anchor 生成任务已启动。"})
            elif path == "/api/stop":
                with LOCK:
                    proc = STATE.get("process")
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    self.send_json({"message": "已发送停止信号。"})
                else:
                    self.send_json({"message": "当前没有运行中的任务。"})
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)


def main():
    parser = argparse.ArgumentParser(description="EACL remote training dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print("在 VS Code Remote 的 Ports 面板转发这个端口即可在本机浏览器打开。")
    server.serve_forever()


if __name__ == "__main__":
    main()
