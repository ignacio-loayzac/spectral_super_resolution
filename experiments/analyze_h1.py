#!/usr/bin/env python
"""
Preregistered analysis of H1/H5/H2 (experiments/PREREGISTRO_H1.md).

  python experiments/analyze_h1.py runs   results/runpod_phase4_final_<ts>.json
      -> experiments/eval_h1_runs.json (15 single models, FINAL checkpoints) and
         experiments/eval_h1_ensembles.json (one 3-seed ensemble per arm)
  python experiments/analyze_h1.py report
      -> results/h1_report.md / .json from results/eval_h1.json (+ eval_h1_ensembles.json, eval_2020.json)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ARMS = ["M0", "C", "P", "A", "AC"]
DESC = {"M0": "MSE, 175 bandas", "C": "MSE + consistencia", "P": "PCA-16 (decodificador lineal congelado)",
        "A": "AE-16 (decodificador no lineal congelado)", "AC": "AE-16 + consistencia"}
CONTRASTS = [("A", "M0", "H1"), ("P", "M0", "H5"), ("A", "P", "H1 vs control"), ("C", "M0", "H2"), ("AC", "A", "H2 (latente)")]


def runs(record):
    rec = json.loads(Path(record).read_text())
    single, by_arm = [], defaultdict(list)
    for j in rec["jobs"]:
        if j.get("status") != "done":
            continue
        _, arm, seed = j["experiment_name"].split("/")
        spec = {"name": f"{arm} {seed}", "run": j["r2_prefix"],
                "checkpoint": f"r2://{j['r2_prefix']}/checkpoint_epoch_final.pt"}
        single.append(spec)
        by_arm[arm].append({k: v for k, v in spec.items() if k != "name"} | {"name": f"{arm} {seed}"})
    ens = [{"name": f"{arm} ensamble", "ensemble": members} for arm, members in by_arm.items()]
    Path("experiments/eval_h1_runs.json").write_text(json.dumps(single, indent=2))
    Path("experiments/eval_h1_ensembles.json").write_text(json.dumps(ens, indent=2))
    print(f"{len(single)} single runs, {len(ens)} ensembles")


def report():
    ev = json.loads(Path("results/eval_h1.json").read_text())["results"]
    fam = defaultdict(list)
    for r in ev:
        fam[r["name"].split()[0]].append(r)
    agg, tiles = {}, {}
    for a in ARMS:
        rs = fam[a]
        agg[a] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1))) for k in ("rmse", "sam", "ssim", "mae")}
        agg[a]["groups"] = {g: float(np.mean([r["rmse_by_group"][g] for r in rs])) for g in rs[0]["rmse_by_group"]}
        tiles[a] = {m: np.mean([r["per_tile"][m] for r in rs], axis=0) for m in ("rmse", "sam")}
    tests = []
    for x, y, h in CONTRASTS:
        for m in ("rmse", "sam"):
            d = tiles[x][m] - tiles[y][m]
            tests.append({"hyp": h, "contrast": f"{x} vs {y}", "metric": m, "median_diff": float(np.median(d)),
                          "rel_mean_diff": float(agg[x][m][0] / agg[y][m][0] - 1),
                          "share_better": float((d < 0).mean()), "p": float(wilcoxon(tiles[x][m], tiles[y][m]).pvalue)})
    for t, pa in zip(tests, multipletests([t["p"] for t in tests], method="holm")[1]):
        t["p_holm"] = float(pa)
    h1 = {t["metric"]: t for t in tests if t["contrast"] == "A vs M0"}
    confirmed = (h1["rmse"]["median_diff"] < 0 and agg["A"]["rmse"][0] < agg["M0"]["rmse"][0] and h1["rmse"]["p_holm"] < 0.05
                 and not (h1["sam"]["median_diff"] > 0 and h1["sam"]["p_holm"] < 0.05))

    md = ["# Resultado del estudio preregistrado H1 / H5 / H2", "",
          f"**H1 (criterio preregistrado): {'CONFIRMADA' if confirmed else 'NO CONFIRMADA'}**", "",
          "## Test (escenas 11–15, 2 314 tiles, píxeles válidos, reflectancia) — media ± d.e. de 3 semillas", "",
          "| Brazo | Descripción | RMSE | SAM (rad) | SSIM | MAE |", "| :--- | :--- | :---: | :---: | :---: | :---: |"]
    for a in ARMS:
        g = agg[a]
        md.append(f"| {a} | {DESC[a]} | {g['rmse'][0]:.4f} ± {g['rmse'][1]:.4f} | {g['sam'][0]:.4f} ± {g['sam'][1]:.4f} | "
                  f"{g['ssim'][0]:.3f} ± {g['ssim'][1]:.3f} | {g['mae'][0]:.4f} ± {g['mae'][1]:.4f} |")
    md += ["", "## Contrastes preregistrados (Wilcoxon pareado por tile, Holm sobre 10 pruebas)", "",
           "| Hipótesis | Contraste | Métrica | Δ media relativa | Mediana Δ por tile | % tiles mejor | p (Holm) |",
           "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |"]
    for t in tests:
        md.append(f"| {t['hyp']} | {t['contrast']} | {t['metric'].upper()} | {100 * t['rel_mean_diff']:+.1f}% | "
                  f"{t['median_diff']:+.5f} | {100 * t['share_better']:.1f}% | {t['p_holm']:.1e} |")
    md += ["", "## RMSE por grupo de bandas (media de semillas)", "",
           "| Brazo | " + " | ".join(agg["M0"]["groups"]) + " |", "| :--- |" + " :---: |" * len(agg["M0"]["groups"])]
    for a in ARMS:
        md.append(f"| {a} | " + " | ".join(f"{v:.4f}" for v in agg[a]["groups"].values()) + " |")
    if Path("results/eval_h1_ensembles.json").exists():
        en = json.loads(Path("results/eval_h1_ensembles.json").read_text())["results"]
        md += ["", "## Análisis secundario: ensamble de las 3 semillas", "", "| Ensamble | RMSE | SAM | SSIM |", "| :--- | :---: | :---: | :---: |"]
        md += [f"| {r['name']} | {r['rmse']:.4f} | {r['sam']:.4f} | {r['ssim']:.3f} |" for r in en]
    if Path("results/eval_2020.json").exists():
        ref = json.loads(Path("results/eval_2020.json").read_text())["results"]
        md += ["", "## Referencia descriptiva: checkpoints 2020 (otros datos y protocolo)", "", "| Modelo | RMSE | SAM | SSIM |", "| :--- | :---: | :---: | :---: |"]
        md += [f"| {r['name']} | {r['rmse']:.4f} | {r['sam']:.4f} | {r['ssim']:.3f} |" for r in ref]
    Path("results/h1_report.md").write_text("\n".join(md) + "\n")
    Path("results/h1_report.json").write_text(json.dumps({"confirmed_H1": confirmed, "arms": agg, "tests": tests}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    runs(sys.argv[2]) if sys.argv[1] == "runs" else report()
