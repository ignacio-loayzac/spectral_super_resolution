#!/usr/bin/env python
"""Aggregate results/downstream_bg.json per arm (mean ± sd over the 3 seeds) -> results/downstream_bg.md."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ["M0", "C", "P", "A", "AC"]
data = json.loads(Path("results/downstream_bg.json").read_text())
fam = defaultdict(list)
for name, r in data["per_model"].items():
    fam[name.split()[0] if not name.startswith("2020") else "2020"].append(r)


def ms(vals, fmt="{:.3f}"):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "—"
    return (fmt.format(np.mean(vals)) + (" ± " + fmt.format(np.std(vals, ddof=1)) if len(vals) > 1 else ""))


arms = [a for a in ARMS + ["2020"] if a in fam]
out = {"B": {}, "G": {}}
md = [f"# Experimentos downstream B y G (test, {data['pixels']:,} píxeles; vegetación {data['mask_share']['veg']:.0%}, suelo {data['mask_share']['soil']:.0%})", "",
      "Métricas sobre píxeles con espectro real y predicho físicamente válidos (reflectancia > 0.01 en las bandas usadas). Media ± d.e. de 3 semillas.", "",
      "## B · Índices de banda estrecha", ""]
for idx in ("PRI", "REP", "NDSI"):
    md += [f"### {idx}", "", "| Brazo | R² | Spearman ρ | Error abs. mediano | Pendiente | Predicciones inválidas |",
           "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for a in arms:
        rs = fam[a]
        v = [r["B"][idx]["valid"] for r in rs]
        row = {k: [x[k] for x in v if x] for k in ("r2", "spearman", "medae", "slope")}
        inv = [r["B"][idx]["invalid_pred_rate"] for r in rs]
        out["B"].setdefault(idx, {})[a] = {k: (float(np.mean(vv)), float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0) for k, vv in {**row, "invalid": inv}.items()}
        md.append(f"| {a} | {ms(row['r2'])} | {ms(row['spearman'])} | {ms(row['medae'], '{:.4f}')} | {ms(row['slope'])} | {ms([100*x for x in inv], '{:.1f}')}% |")
    md.append("")
md += ["## G · Rasgos de absorción (remoción lineal del continuo)", "",
       "| Rasgo | Brazo | R² profundidad | ρ profundidad | R² área | Error de posición mediano (nm) | Detección | Predicciones inválidas |",
       "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
for feat in ("clorofila", "agua_foliar", "Al-OH", "CO3"):
    for a in arms:
        rs = [r["G"][feat] for r in fam[a]]
        bd = [x["bd"] for x in rs if x["bd"]]
        area = [x["area"] for x in rs if x["area"]]
        vals = {"bd_r2": [x["r2"] for x in bd], "bd_rho": [x["spearman"] for x in bd], "area_r2": [x["r2"] for x in area],
                "pos": [x["pos_err_median_nm"] for x in rs], "det": [x["detection_rate"] for x in rs],
                "invalid": [x["invalid_pred_rate"] for x in rs]}
        out["G"].setdefault(feat, {})[a] = {k: (float(np.nanmean(v)) if v else None) for k, v in vals.items()}
        md.append(f"| {feat} | {a} | {ms(vals['bd_r2'])} | {ms(vals['bd_rho'])} | {ms(vals['area_r2'])} | {ms(vals['pos'], '{:.1f}')} | "
                  f"{ms([100*x for x in vals['det'] if x is not None], '{:.0f}')}% | {ms([100*x for x in vals['invalid']], '{:.1f}')}% |")
Path("results/downstream_bg.md").write_text("\n".join(md) + "\n")
Path("results/downstream_bg_summary.json").write_text(json.dumps(out, indent=2))
print("\n".join(md))
