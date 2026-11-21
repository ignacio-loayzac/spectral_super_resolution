#!/usr/bin/env python
"""
Aggregate the phase-2 test evaluation (BACKLOG T-019 / T-043).

Input: results/eval_final.json (scripts/evaluate_models.py over the 12 final runs)
       and results/eval_2020.json (reference rows).
Output: results/final_summary.md and results/final_summary.json with
  - mean ± sd over seeds of RMSE, SAM, SSIM, MAE (test, masked, reflectance)
  - paired Wilcoxon signed-rank tests per test tile (tile metric averaged over the
    3 seeds of each variant) for every variant vs M0, Holm-corrected, with the
    median paired difference and the share of tiles that improve.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

DESC = {"M0": "MSE (con mejoras A)", "M1": "Perceptual 2020 corregida (AE espacial, Gram bruta)",
        "M2": "Perceptual mejorada (AE espectral, Gram de correlación, features)", "M3": "M2 + consistencia espectral"}

ev = json.loads(Path("results/eval_final.json").read_text())["results"]
fam = defaultdict(list)
for r in ev:
    fam[r["name"].split()[0]].append(r)

summary, tiles = {}, {}
for f, rs in sorted(fam.items()):
    summary[f] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1)) if len(rs) > 1 else 0.0)
                  for k in ("rmse", "sam", "ssim", "mae")}
    summary[f]["n_seeds"] = len(rs)
    summary[f]["rmse_by_group"] = {g: float(np.mean([r["rmse_by_group"][g] for r in rs])) for g in rs[0]["rmse_by_group"]}
    tiles[f] = {m: np.mean([r["per_tile"][m] for r in rs], axis=0) for m in ("rmse", "sam")}

tests = []
for f in sorted(fam):
    if f == "M0":
        continue
    for m in ("rmse", "sam"):
        d = tiles[f][m] - tiles["M0"][m]
        stat, p = wilcoxon(tiles[f][m], tiles["M0"][m])
        tests.append({"variant": f, "metric": m, "median_diff": float(np.median(d)), "p": float(p),
                      "share_tiles_better": float((d < 0).mean())})
if tests:
    adj = multipletests([t["p"] for t in tests], method="holm")[1]
    for t, pa in zip(tests, adj):
        t["p_holm"] = float(pa)

ref = json.loads(Path("results/eval_2020.json").read_text())["results"]
md = ["## Test (escenas 11–15, 2 314 tiles, píxeles válidos, reflectancia)", "",
      "| Variante | Descripción | RMSE | SAM (rad) | SSIM | MAE | Semillas |", "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"]
for f, s in summary.items():
    md.append(f"| {f} | {DESC.get(f, f)} | {s['rmse'][0]:.4f} ± {s['rmse'][1]:.4f} | {s['sam'][0]:.4f} ± {s['sam'][1]:.4f} | "
              f"{s['ssim'][0]:.3f} ± {s['ssim'][1]:.3f} | {s['mae'][0]:.4f} ± {s['mae'][1]:.4f} | {s['n_seeds']} |")
for r in ref:
    md.append(f"| ref. | {r['name']} (checkpoint 2020) | {r['rmse']:.4f} | {r['sam']:.4f} | {r['ssim']:.3f} | {r['mae']:.4f} | 1 |")
md += ["", "## Pruebas pareadas por tile vs M0 (Wilcoxon, corrección de Holm)", "",
       "| Variante | Métrica | Mediana de la diferencia (variante − M0) | % tiles que mejoran | p (Holm) |",
       "| :--- | :--- | :---: | :---: | :---: |"]
for t in tests:
    md.append(f"| {t['variant']} | {t['metric'].upper()} | {t['median_diff']:+.5f} | {100 * t['share_tiles_better']:.1f}% | {t['p_holm']:.2e} |")
md += ["", "## RMSE por grupo de bandas (media de semillas)", "",
       "| Variante | " + " | ".join(next(iter(summary.values()))["rmse_by_group"]) + " |",
       "| :--- |" + " :---: |" * len(next(iter(summary.values()))["rmse_by_group"])]
for f, s in summary.items():
    md.append(f"| {f} | " + " | ".join(f"{v:.4f}" for v in s["rmse_by_group"].values()) + " |")
Path("results/final_summary.md").write_text("\n".join(md) + "\n")
Path("results/final_summary.json").write_text(json.dumps({"summary": summary, "tests": tests}, indent=2))
print("\n".join(md))
