#!/usr/bin/env python
"""Build the evaluate_models.py spec list from a phase-2 RunPod record (one entry per finished run)."""
import json
import sys
from pathlib import Path

rec = json.loads(Path(sys.argv[1]).read_text())
runs = [{"name": f"{j['experiment_name'].split('/')[1]} {j['experiment_name'].split('/')[2]}", "run": j["r2_prefix"]}
        for j in rec["jobs"] if j.get("status") == "done" and j.get("r2_prefix")]
Path("experiments/eval_final_runs.json").write_text(json.dumps(runs, indent=2))
print(f"{len(runs)} runs -> experiments/eval_final_runs.json")
