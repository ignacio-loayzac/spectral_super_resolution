#!/usr/bin/env python
"""
Summarise the phase-1 validation sweep and write the phase-2 plan (BACKLOG E-M).

Selection rule (fixed before looking at test data): within each family (M1, M2, M3)
pick the configuration with the lowest validation score
    score = RMSE_val / RMSE_val(M0) + SAM_val / SAM_val(M0)
using each run's best epoch. Phase 2 retrains M0 and the three winners with
seeds 1, 2, 3 (50 epochs max, early stopping patience 8). M3 = winning M2
configuration + the consistency coefficient of the best M3 sweep run.

Usage: python experiments/select_and_plan.py results/runpod_phase1_sweep_<ts>.json
"""

import json
import sys
from pathlib import Path

rec = json.loads(Path(sys.argv[1]).read_text())
rows = []
for j in rec["jobs"]:
    b = j.get("best") or {}
    if j.get("status") != "done" or not b:
        print(f"skip {j['experiment_name']}: {j.get('status')}")
        continue
    rows.append({"name": j["experiment_name"].split("/")[-1], "request": j["request"], "best_epoch": b["epoch"],
                 "epochs_run": j.get("epochs_run"), "rmse": b["val_rmse_reflectance"], "sam": b["val_sam"],
                 "ssim": b["val_ssim"], "r2_prefix": j.get("r2_prefix")})
m0 = next(r for r in rows if r["name"] == "M0_mse")
for r in rows:
    r["score"] = r["rmse"] / m0["rmse"] + r["sam"] / m0["sam"]

lines = ["| Config | β | γ | λ | Mejor época | RMSE val | SAM val (rad) | SSIM val | Score |",
         "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for r in sorted(rows, key=lambda r: r["name"]):
    q = r["request"]
    lines.append(f"| {r['name']} | {q.get('style_loss_coeff', 0):g} | {q.get('feature_loss_coeff', 0):g} | "
                 f"{q.get('consistency_coeff', 0):g} | {r['best_epoch']}/{r['epochs_run']} | {r['rmse']:.4f} | "
                 f"{r['sam']:.4f} | {r['ssim']:.3f} | {r['score']:.3f} |")
table = "\n".join(lines)
print(table)

winners = {fam: min((r for r in rows if r["name"].startswith(fam)), key=lambda r: r["score"], default=None)
           for fam in ("M1", "M2", "M3")}
print("\nwinners:", {k: (v["name"] if v else None) for k, v in winners.items()})

finals = {"M0": m0["request"]}
for fam in ("M1", "M2"):
    if winners[fam]:
        finals[fam] = winners[fam]["request"]
if winners["M2"] and winners["M3"]:
    m3 = dict(winners["M2"]["request"])
    for k in ("consistency_coeff", "degradation_path", "consistency_bands"):
        m3[k] = winners["M3"]["request"][k]
    finals["M3"] = m3

plan = []
for fam, req in finals.items():
    for seed in (1, 2, 3):
        body = dict(req)
        body.update(seed=seed, epochs=50, early_stopping_patience=8, experiment_name=f"final/{fam}/seed{seed}")
        plan.append(body)
Path("experiments/phase2_final.json").write_text(json.dumps(plan, indent=2))
Path("results/phase1_sweep_summary.md").write_text(
    "# Fase 1 — barrido en validación (escena 2)\n\n"
    "Regla de selección: score = RMSE/RMSE(M0) + SAM/SAM(M0) en la mejor época de validación.\n\n"
    + table + "\n\nGanadores: " + ", ".join(f"{k}: {v['name']}" for k, v in winners.items() if v) + "\n")
print(f"\nphase2: {len(plan)} jobs -> experiments/phase2_final.json")
