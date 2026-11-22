#!/usr/bin/env python
"""Training-curve figures for the thesis (Studies 2 and 3), from results/training_histories/.

Main text:
  fig_val_rmse_estudio2.pdf      validation RMSE per epoch, M0-M3 (validation scene 2), 3 seeds each
  fig_val_rmse_estudio3_cv.pdf   validation RMSE per epoch in the scene cross-validation, one panel per fold
Appendix:
  fig_ap_perdidas_estudio2.pdf   train / validation loss per epoch, one panel per variant
  fig_ap_componentes_m1.pdf      the two terms of the M1 training loss
  fig_ap_perdidas_estudio3_cv.pdf  train / validation loss per epoch in the cross-validation, one panel per arm
  fig_ap_perdidas_estudio3_final.pdf  training loss of the final Study 3 models (fixed epochs, no validation)
Colour = variant / arm; line style = seed or fold. Markers at the last epoch: x = epoch limit,
o = early stopping.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

H = Path("results/training_histories")
OUT = Path("../Masters_Thesis_DI/figures")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
S2 = {"M0": ("M0: MSE", "#7f7f7f"), "M1": ("M1: Gram bruta", "#cb181d"),
      "M2": ("M2: perceptual mejorada", "#fdae6b"), "M3": ("M3: M2 + consistencia", "#08519c")}
S3 = {"M0": ("M0 (MSE)", "#7f7f7f"), "C": ("C (+ consistencia)", "#41ab5d"), "P": ("P (PCA-16)", "#fdae6b"),
      "A": ("A (AE-16)", "#6baed6"), "AC": ("AC (AE-16 + consistencia)", "#08519c")}
SEED_LS = {1: "-", 2: "--", 3: ":"}
FOLD_LS = {0: "-", 1: "--", 2: ":", 3: "-."}
FOLDS = {0: "{1, 2}", 1: "{3, 4}", 2: "{7, 8}", 3: "{9, 10}"}


def h(name):
    return json.loads((H / f"{name}.json").read_text())


def curve(ax, y, col, ls, stop=None, lw=1.3):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ep = range(1, len(y) + 1)
    ax.plot(ep, y, color=col, ls=ls, lw=lw)
    if stop is not None:
        early = stop.startswith("early")
        ax.plot(len(y), y[-1], "o" if early else "x", color=col, ms=5, mfc="none" if early else col)


def style_legend(ax, styles, title, loc="upper right", extra=(), anchor=None):
    hs = [Line2D([], [], color="k", ls=ls, lw=1.2, label=lab) for lab, ls in styles] + list(extra)
    return ax.legend(handles=hs, title=title, frameon=False, fontsize=7.5, title_fontsize=8, loc=loc,
                     bbox_to_anchor=anchor, ncol=2 if anchor else 1)


def color_legend(ax, spec, loc):
    hs = [Line2D([], [], color=c, lw=2, label=lab) for lab, c in spec.values()]
    return ax.legend(handles=hs, frameon=False, fontsize=8, loc=loc)


def s2(v, s):
    """Study 2 run: for M1 the extended run (identical to the 50-epoch run for its first 50 epochs)."""
    return h(f"long__M1__seed{s}") if v == "M1" else h(f"final__{v}__seed{s}")


LIM50 = Line2D([], [], color="#7f7f7f", lw=0.9, ls=(0, (4, 3)), label="límite del Estudio 2 (50 épocas)")

END = [Line2D([], [], color="k", marker="x", ls="none", label="límite de épocas"),
       Line2D([], [], color="k", marker="o", mfc="none", ls="none", label="parada temprana")]

# ---------------- main text: Study 2 validation RMSE ----------------
fig, (ax, az) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1]})
for v, (lab, col) in S2.items():
    for s in (1, 2, 3):
        r = s2(v, s)
        for a in (ax, az):
            curve(a, r["val_rmse_reflectance"], col, SEED_LS[s], r["stop_reason"])
ax.set_yscale("log"); ax.set_title("(a) Todas las variantes (escala logarítmica)", fontsize=9.5, loc="left")
az.set_ylim(0.028, 0.075); az.set_title("(b) Ampliación del rango de M0, M2 y M3 (escala lineal)", fontsize=9.5, loc="left")
for a in (ax, az):
    a.set_xlabel("Época"); a.set_ylabel("RMSE de validación (reflectancia)")
ax.set_ylim(0.02, 3000)
ax.axvline(50, color="#7f7f7f", lw=0.9, ls=(0, (4, 3)))
ax.add_artist(color_legend(ax, S2, "upper right"))
style_legend(ax, [(f"semilla {s}", SEED_LS[s]) for s in (1, 2, 3)], None, loc="center", extra=END[1:] + [LIM50], anchor=(0.6, 0.42))
az.set_xlim(0, 41)
fig.tight_layout(); fig.savefig(OUT / "fig_val_rmse_estudio2.pdf"); plt.close(fig)

# ---------------- main text: Study 3 CV validation RMSE ----------------
fig, axs = plt.subplots(1, 4, figsize=(11, 3.3), sharey=True)
for f, ax in enumerate(axs):
    for a, (lab, col) in S3.items():
        r = h(f"h1cv__{a}__fold{f}")
        curve(ax, r["val_rmse_reflectance"], col, "-", r["stop_reason"])
    ax.set_title(f"Validación: escenas {FOLDS[f]}", fontsize=9.5)
    ax.set_xlabel("Época")
axs[0].set_ylabel("RMSE de validación (reflectancia)")
axs[0].set_yscale("log")
fig.legend(handles=[Line2D([], [], color=c, lw=2, label=lab) for lab, c in S3.values()] + END,
           loc="lower center", ncol=7, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "fig_val_rmse_estudio3_cv.pdf"); plt.close(fig)

# ---------------- appendix: Study 2 losses ----------------
TR, VA = "#252525", "#e6550d"
fig, axs = plt.subplots(2, 2, figsize=(10, 7))
for ax, (v, (lab, _)) in zip(axs.flat, S2.items()):
    for s in (1, 2, 3):
        r = s2(v, s)
        curve(ax, r["train_losses"], TR, SEED_LS[s])
        curve(ax, r["val_losses"], VA, SEED_LS[s], r["stop_reason"])
    ax.set_yscale("log"); ax.set_title(lab, fontsize=10, loc="left"); ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
    if v == "M1":
        ax.axvline(50, color="#7f7f7f", lw=0.9, ls=(0, (4, 3)))
style_legend(axs[0, 0], [(f"semilla {s}", SEED_LS[s]) for s in (1, 2, 3)], None, loc="upper right",
             extra=[Line2D([], [], color=TR, lw=2, label="entrenamiento"), Line2D([], [], color=VA, lw=2, label="validación")] + END[1:] + [LIM50])
fig.tight_layout(); fig.savefig(OUT / "fig_ap_perdidas_estudio2.pdf"); plt.close(fig)

# ---------------- appendix: M1 loss components ----------------
fig, ax = plt.subplots(figsize=(8, 3.8))
for s in (1, 2, 3):
    c = h(f"long__M1__seed{s}")["train_loss_components"]
    curve(ax, [e["content"] for e in c], "#7f7f7f", SEED_LS[s])
    curve(ax, [e["style"] for e in c], "#cb181d", SEED_LS[s])
ax.set_yscale("log"); ax.set_xlabel("Época"); ax.set_ylabel("Valor medio por época (entrenamiento)")
ax.axvline(50, color="#7f7f7f", lw=0.9, ls=(0, (4, 3)))
style_legend(ax, [(f"semilla {s}", SEED_LS[s]) for s in (1, 2, 3)], None, loc="upper right",
             extra=[Line2D([], [], color="#7f7f7f", lw=2, label=r"$\mathcal{L}_{content}$ (MSE)"),
                    Line2D([], [], color="#cb181d", lw=2, label=r"$\beta\,\mathcal{L}_{Gram}$"), LIM50])
fig.tight_layout(); fig.savefig(OUT / "fig_ap_componentes_m1.pdf"); plt.close(fig)

# ---------------- appendix: Study 3 CV losses ----------------
fig, axs = plt.subplots(2, 3, figsize=(11, 6.6))
for ax, (a, (lab, _)) in zip(axs.flat, S3.items()):
    for f in range(4):
        r = h(f"h1cv__{a}__fold{f}")
        curve(ax, r["train_losses"], TR, FOLD_LS[f])
        curve(ax, r["val_losses"], VA, FOLD_LS[f], r["stop_reason"])
    ax.set_yscale("log"); ax.set_title(lab, fontsize=10, loc="left"); ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
axs.flat[-1].axis("off")
style_legend(axs.flat[-1], [(f"validación en escenas {FOLDS[f]}", FOLD_LS[f]) for f in range(4)], None, loc="center",
             extra=[Line2D([], [], color=TR, lw=2, label="entrenamiento"), Line2D([], [], color=VA, lw=2, label="validación")] + END)
fig.tight_layout(); fig.savefig(OUT / "fig_ap_perdidas_estudio3_cv.pdf"); plt.close(fig)

# ---------------- appendix: Study 3 final models (training loss only) ----------------
fig, ax = plt.subplots(figsize=(8, 3.8))
for a, (lab, col) in S3.items():
    for s in (1, 2, 3):
        curve(ax, h(f"h1final__{a}__seed{s}")["train_losses"], col, SEED_LS[s])
ax.set_yscale("log"); ax.set_xlabel("Época"); ax.set_ylabel("Pérdida de entrenamiento")
ax.add_artist(color_legend(ax, S3, "upper right"))
style_legend(ax, [(f"semilla {s}", SEED_LS[s]) for s in (1, 2, 3)], None, loc="center right")
fig.tight_layout(); fig.savefig(OUT / "fig_ap_perdidas_estudio3_final.pdf"); plt.close(fig)
print("figures written")

# ---------------- appendix: extended M1 training (up to 300 epochs) ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
for s in (1, 2, 3):
    r = h(f"long__M1__seed{s}")
    curve(a, r["val_rmse_reflectance"], "#cb181d", SEED_LS[s], r["stop_reason"])
    curve(b, r["train_losses"], TR, SEED_LS[s])
    curve(b, r["val_losses"], VA, SEED_LS[s], r["stop_reason"])
for ax in (a, b):
    ax.axvline(50, color="#7f7f7f", lw=0.8, ls=(0, (4, 3)))
    ax.set_yscale("log"); ax.set_xlabel("Época")
a.axhline(0.0338, color="#7f7f7f", lw=1.0)
a.text(2, 0.0338 * 1.25, "M0 (mejor validación)", fontsize=8, color="#525252")
a.text(51, 300, "límite original\n(50 épocas)", fontsize=8, color="#525252", va="top")
a.set_ylabel("RMSE de validación (reflectancia)"); b.set_ylabel("Pérdida")
a.set_title("(a) RMSE de validación de M1", fontsize=9.5, loc="left")
b.set_title("(b) Pérdida de entrenamiento y validación de M1", fontsize=9.5, loc="left")
style_legend(a, [(f"semilla {s}", SEED_LS[s]) for s in (1, 2, 3)], None, loc="upper right", extra=END[1:])
style_legend(b, [(f"semilla {s}", SEED_LS[s]) for s in (1, 2, 3)], None, loc="upper right",
             extra=[Line2D([], [], color=TR, lw=2, label="entrenamiento"), Line2D([], [], color=VA, lw=2, label="validación")])
fig.tight_layout(); fig.savefig(OUT / "fig_ap_m1_largo.pdf"); plt.close(fig)
print("long M1 figure written")
