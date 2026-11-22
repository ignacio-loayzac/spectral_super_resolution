#!/usr/bin/env python
"""Appendix figure: first epoch of M1, batch by batch (results/diag_m1_first_epoch.json).

(a) RMS of the prediction over valid pixels (reflectance) vs batch, with the target RMS;
(b) ratio between the norms of the gradients of the Gram term and of the content term.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.loads(Path("results/diag_m1_first_epoch.json").read_text())
OUT = Path("../Masters_Thesis_DI/figures")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = {"1": "#cb181d", "2": "#fb6a4a", "3": "#08519c", "42": "#6baed6"}
LAB = {"1": "semilla 1 (escapa)", "2": "semilla 2 (escapa)", "3": "semilla 3 (se recupera)", "42": "semilla 42 (se recupera)"}

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
for s, rows in D.items():
    x = [r["batch"] for r in rows]
    a.plot(x, [r["pred_rms"] for r in rows], color=COL[s], lw=1.3, label=LAB[s])
    b.plot(x, [r["grad_style"] / r["grad_content"] for r in rows], color=COL[s], lw=1.0, alpha=0.9)
tr = sum(r["target_rms"] for rows in D.values() for r in rows) / sum(len(rows) for rows in D.values())
a.axhline(tr, color="k", lw=0.9, ls="--", label="escala del objetivo (Hyperion)")
b.axhline(1, color="k", lw=0.9, ls="--")
for ax in (a, b):
    ax.set_yscale("log"); ax.set_xlabel("Lote (primera época)")
a.set_ylabel("RMS de la predicción (reflectancia)")
b.set_ylabel(r"$\|\nabla_\theta\,\beta\mathcal{L}_{Gram}\| \,/\, \|\nabla_\theta\,\mathcal{L}_{content}\|$")
a.set_title("(a) Escala de la salida", fontsize=9.5, loc="left")
b.set_title("(b) Gradiente de Gram relativo al de contenido", fontsize=9.5, loc="left")
a.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_ap_m1_diag.pdf")
print("written")
