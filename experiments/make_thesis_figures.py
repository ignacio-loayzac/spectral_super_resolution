#!/usr/bin/env python
"""Figures for the thesis (results chapter) from the cached downstream spectra and results/eval_h1.json.

  fig_h1_rmse_por_grupo.pdf   RMSE per OLI band group, 5 arms (mean ± sd over 3 seeds)
  fig_h1_espectros.pdf        example test spectra: real vs M0 vs AC (seed 1), fixed random pixels
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("../Masters_Thesis_DI/figures")
ARMS = ["M0", "C", "P", "A", "AC"]
LABEL = {"M0": "M0 (MSE)", "C": "C (+ consistencia)", "P": "P (PCA-16)", "A": "A (AE-16)", "AC": "AC (AE-16 + consistencia)"}
COLOR = {"M0": "#7f7f7f", "C": "#41ab5d", "P": "#fdae6b", "A": "#6baed6", "AC": "#08519c"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# --- RMSE per band group ------------------------------------------------------
ev = json.loads(Path("results/eval_h1.json").read_text())["results"]
fam = defaultdict(list)
for r in ev:
    fam[r["name"].split()[0]].append(r["rmse_by_group"])
groups = list(fam["M0"][0].keys())
x = np.arange(len(groups))
w = 0.16
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, a in enumerate(ARMS):
    vals = np.array([[g[k] for k in groups] for g in fam[a]])
    ax.bar(x + (i - 2) * w, vals.mean(0), w, yerr=vals.std(0, ddof=1), label=LABEL[a], color=COLOR[a],
           error_kw={"elinewidth": 0.8, "capsize": 2})
ax.set_xticks(x, ["Costera" if g == "CA" else "Azul" if g == "Blue" else "Verde" if g == "Green" else "Rojo" if g == "Red" else g for g in groups])
ax.set_ylabel("RMSE (reflectancia)")
ax.set_ylim(0, 0.125)
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_h1_rmse_por_grupo.pdf")

# --- Example spectra ----------------------------------------------------------
cache = Path("outputs/downstream_cache")
real = np.load(cache / "_real.npy")
m0, ac = np.load(cache / "M0_seed1.npy"), np.load(cache / "AC_seed1.npy")
wl = np.load("data/wavelengths_175.npy")
nir, red = np.argmin(np.abs(wl - 800)), np.argmin(np.abs(wl - 670))
ndvi = (real[:, nir] - real[:, red]) / (real[:, nir] + real[:, red] + 1e-9)
rng = np.random.default_rng(7)
picks = [rng.choice(np.flatnonzero(ndvi > 0.5)), rng.choice(np.flatnonzero((ndvi > 0.2) & (ndvi < 0.4))),
         rng.choice(np.flatnonzero((ndvi < 0.15) & (ndvi > 0))), rng.choice(np.flatnonzero(ndvi < 0.15))]
titles = ["Vegetación densa", "Vegetación rala", "Suelo", "Suelo / urbano"]
fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), sharey=False)
gaps = np.flatnonzero(np.diff(wl) > 20)  # removed water-absorption windows
for ax, p, t in zip(axs, picks, titles):
    for spec, lab, c, lw in [(real[p], "Hyperion real", "black", 1.4), (m0[p], "M0", COLOR["M0"], 1.1), (ac[p], "AC", COLOR["AC"], 1.1)]:
        s = spec.astype(float).copy()
        s_plot = np.insert(s, gaps + 1, np.nan)
        w_plot = np.insert(wl, gaps + 1, np.nan)
        ax.plot(w_plot, s_plot, color=c, lw=lw, label=lab)
    ax.set_title(t, fontsize=10)
    ax.set_xlabel("Longitud de onda (nm)")
axs[0].set_ylabel("Reflectancia")
axs[0].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_h1_espectros.pdf")
print("figuras escritas en", OUT.resolve())
