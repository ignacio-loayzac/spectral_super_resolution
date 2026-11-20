#!/usr/bin/env python
"""
Run a queue of training jobs on ONE RunPod GPU pod and always delete the pod.

  python scripts/runpod_experiments.py --plan experiments/phase1.json --max-usd 6

The plan is a JSON list of /train request bodies (see spectral_sr.api.TrainRequest).
The pod runs the published image, downloads data once into its cache (T-018), runs
the jobs sequentially (the API serialises GPU jobs) and uploads every run to the
R2 experiments bucket. This script:

  1. creates the pod (first GPU type with stock from --gpus), passing the R2 and
     API credentials from .env as environment variables;
  2. waits for /health, submits every job, polls phases/epochs;
  3. stops early if the estimated cost would exceed --max-usd;
  4. deletes the pod in a ``finally`` block (also on Ctrl-C or errors);
  5. writes a local log (results/runpod_<plan>_<timestamp>.json) with job ids,
     R2 prefixes, final metrics, wall time and estimated cost.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ENV_KEYS = ["SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID", "SPECTRAL_RECONSTRUCTION_R2_SECRET_ACCESS_KEY",
            "SPECTRAL_RECONSTRUCTION_R2_ENDPOINT_URL", "SPECTRAL_RECONSTRUCTION_R2_BUCKET_NAME",
            "SPECTRAL_RECONSTRUCTION_R2_BUCKET_NAME_EXPERIMENTS", "SPECTRAL_RECONSTRUCTION_API_KEY"]
DEFAULT_GPUS = ["NVIDIA GeForce RTX 4090|SECURE", "NVIDIA GeForce RTX 3090|SECURE",
                "NVIDIA RTX A5000|SECURE", "NVIDIA GeForce RTX 4090|COMMUNITY",
                "NVIDIA GeForce RTX 3090|COMMUNITY", "NVIDIA L40S|SECURE"]


def load_env(path=".env"):
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def rp(*args):
    out = subprocess.run(["runpodctl", *args], capture_output=True, text=True,
                         env={**os.environ, "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"]})
    try:
        return json.loads(out.stdout or out.stderr)
    except json.JSONDecodeError:
        return {"error": (out.stdout + out.stderr)[:500]}


def http(method, url, body=None, timeout=30):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {os.environ['SPECTRAL_RECONSTRUCTION_API_KEY']}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--image", default="ghcr.io/ignacio-loayzac/spectral_super_resolution:1.2.0")
    ap.add_argument("--gpus", nargs="*", default=DEFAULT_GPUS, help='"<gpu id>|SECURE|COMMUNITY" in order of preference')
    ap.add_argument("--max-usd", type=float, required=True, help="hard budget for this pod")
    ap.add_argument("--name", default="spectral-exp")
    ap.add_argument("--disk-gb", type=int, default=80)
    args = ap.parse_args()
    load_env()
    plan = json.loads(Path(args.plan).read_text())
    env = {k: os.environ[k] for k in ENV_KEYS}
    env.update(SPECTRAL_RECONSTRUCTION_PORT="8000", OMP_NUM_THREADS="4",
               SPECTRAL_RECONSTRUCTION_CACHE_DIR="/tmp/spectral_sr_cache")

    record = {"plan": args.plan, "image": args.image, "started_at": datetime.now(timezone.utc).isoformat(), "jobs": []}
    out = Path("results") / f"runpod_{Path(args.plan).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    pod_id, cost_hr, t_start = None, None, time.time()
    try:
        for spec in args.gpus:
            gpu, cloud = spec.split("|")
            r = rp("pod", "create", "--name", args.name, "--image", args.image, "--gpu-id", gpu,
                   "--cloud-type", cloud, "--container-disk-in-gb", str(args.disk_gb), "--ports", "8000/http",
                   "--env", json.dumps(env), "--ssh=false")
            if r.get("id"):
                pod_id, cost_hr = r["id"], float(r.get("costPerHr") or 0)
                record.update(pod_id=pod_id, gpu=gpu, cloud=cloud, cost_per_hr=cost_hr)
                log(f"pod {pod_id} on {gpu} ({cloud}) at {cost_hr} USD/h")
                break
            log(f"no stock: {gpu} {cloud}")
        if not pod_id:
            raise SystemExit("no GPU available")
        if cost_hr <= 0:
            raise SystemExit("could not read the pod price; aborting for safety")

        def spent():
            return (time.time() - t_start) / 3600 * cost_hr

        url = f"https://{pod_id}-8000.proxy.runpod.net"
        while True:
            try:
                h = http("GET", f"{url}/health", timeout=10)
                if h.get("status") == "ok":
                    log(f"API ready: {h.get('device_name')} (v{h.get('version')})")
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
                pass
            if spent() > args.max_usd:
                raise SystemExit("budget exhausted while waiting for the pod")
            time.sleep(15)

        for body in plan:
            j = http("POST", f"{url}/train", body)
            record["jobs"].append({"job_id": j["job_id"], "experiment_name": body.get("experiment_name"), "request": body})
            log(f"queued {body.get('experiment_name')} -> {j['job_id'][:8]}")

        pending = {j["job_id"] for j in record["jobs"]}
        last_phase = {}
        while pending:
            time.sleep(30)
            for job in record["jobs"]:
                if job["job_id"] not in pending:
                    continue
                try:
                    s = http("GET", f"{url}/jobs/{job['job_id']}")
                except Exception as exc:  # noqa: BLE001
                    log(f"poll error {exc}")
                    continue
                ph = s.get("phase")
                if ph != last_phase.get(job["job_id"]):
                    last_phase[job["job_id"]] = ph
                    extra = ""
                    m = s.get("metrics") or {}
                    if m.get("val_rmse_reflectance"):
                        extra = (f" | val RMSE {m['val_rmse_reflectance'][-1]:.4f} SAM {m['val_sam'][-1]:.3f}"
                                 f" | {m['epoch_seconds'][-1]:.0f}s/epoch")
                    log(f"{job['experiment_name']}: {ph}{extra} | spent ~{spent():.2f} USD")
                if s["status"] in ("done", "failed"):
                    pending.discard(job["job_id"])
                    m = s.get("metrics") or {}
                    job.update(status=s["status"], r2_prefix=s.get("r2_prefix"), error=(s.get("error") or "")[-2000:],
                               best=m.get("best"), stop_reason=m.get("stop_reason"),
                               epochs_run=len(m.get("val_losses", [])), epoch_seconds=m.get("epoch_seconds"),
                               started_at=s.get("started_at"), finished_at=s.get("finished_at"))
                    log(f"{job['experiment_name']}: {s['status'].upper()} -> r2://{s.get('r2_prefix')}"
                        + (f" | best epoch {m['best']['epoch']}" if m.get("best") else "")
                        + (f"\n{job['error'][-600:]}" if s['status'] == 'failed' else ""))
                out.write_text(json.dumps(record, indent=2))
            if spent() > args.max_usd:
                log(f"BUDGET LIMIT {args.max_usd} USD reached: stopping with {len(pending)} jobs unfinished")
                break
    finally:
        if pod_id:
            r = rp("pod", "delete", pod_id)
            log(f"pod {pod_id} deleted: {r}")
        record.update(finished_at=datetime.now(timezone.utc).isoformat(),
                      wall_hours=round((time.time() - t_start) / 3600, 3),
                      estimated_usd=round((time.time() - t_start) / 3600 * (cost_hr or 0), 2))
        out.write_text(json.dumps(record, indent=2))
        log(f"log: {out} | estimated cost {record['estimated_usd']} USD")


if __name__ == "__main__":
    sys.exit(main())
