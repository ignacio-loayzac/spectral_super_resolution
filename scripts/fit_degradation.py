#!/usr/bin/env python
"""
Fit the affine degradation used by the spectral consistency loss (BACKLOG T-088):
Hyperion reflectance Y (N x 175) -> OLI reflectance X (N x 7), X ≈ [Y, 1]·D with
D of shape (176, 7) (last row = per-band intercept), on valid pixels of the
QC-filtered TRAIN split (splits/v2, BACKLOG T-091).

Two candidates are fitted and the one with the higher mean validation R² is kept:
  - "boxcar_affine": nominal OLI band edges (boxcar SRF) + per-band gain/offset
  - "ridge": unconstrained ridge regression with intercept
A non-intercept fit is not used: the sensors show a large scene-dependent offset
in the VNIR (see data/qc_scenes.csv).

Outputs data/degradation_175x7.npy (shape 176x7), data/degradation_fit.json, and
uploads both to r2://metadata/.

Usage: python scripts/fit_degradation.py [--tiles 800] [--alpha 1e-3]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from spectral_sr.storage import R2Config, list_prefix, resolve_directory

SPLIT = "splits/v2"   # train; validation is always splits/v1/*_val_npy
VAL_SPLIT = "splits/v1"


def load_pairs(names, hyp_dir, ls_dir, max_px=None, seed=0):
    Y = np.concatenate([np.load(Path(hyp_dir) / n).reshape(-1, 175) for n in names]).astype(np.float64) / 1e4
    X = np.concatenate([np.load(Path(ls_dir) / n).reshape(-1, 7) for n in names]).astype(np.float64) / 1e4
    valid = (np.abs(Y).sum(1) > 1e-6) & (X != 0).all(1)
    Y, X = Y[valid], X[valid]
    if max_px and len(Y) > max_px:
        idx = np.random.default_rng(seed).choice(len(Y), max_px, replace=False)
        Y, X = Y[idx], X[idx]
    return Y, X


def r2_scores(X, Xh):
    return [float(1 - ((X[:, k] - Xh[:, k]) ** 2).sum() / ((X[:, k] - X[:, k].mean()) ** 2).sum()) for k in range(7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=800, help="random train tiles used for the fit")
    ap.add_argument("--alpha", type=float, default=1e-3, help="ridge strength (reflectance units)")
    ap.add_argument("--max-px", type=int, default=1_000_000)
    args = ap.parse_args()
    cfg = R2Config.from_env()

    train_names = sorted(k.split("/")[-1] for k, _ in list_prefix(f"{SPLIT}/hyperion_train_npy/", cfg))
    names = sorted(random.Random(0).sample(train_names, min(args.tiles, len(train_names))))
    # download only the chosen tiles: copy them into a local folder via the cache of the full prefix
    hyp_dir = resolve_directory(f"{SPLIT}/hyperion_train_npy")  # cached locally
    ls_dir = resolve_directory(f"{SPLIT}/landsat_train_npy")
    Y, X = load_pairs(names, hyp_dir, ls_dir, args.max_px)
    print(f"fit pixels: {len(Y):,} from {len(names)} train tiles")

    hv = resolve_directory(f"{VAL_SPLIT}/hyperion_val_npy")
    lv = resolve_directory(f"{VAL_SPLIT}/landsat_val_npy")
    val_names = sorted(p.name for p in Path(hv).glob("*.npy"))
    Yv, Xv = load_pairs(val_names, hv, lv, args.max_px)
    wl = np.load("data/wavelengths_175.npy")

    candidates = {}
    # (a) boxcar SRF + per-band affine
    edges = [(433, 453), (450, 515), (525, 600), (630, 680), (845, 885), (1560, 1660), (2100, 2300)]
    D = np.zeros((176, 7))
    for k, (lo, hi) in enumerate(edges):
        sel = (wl >= lo) & (wl <= hi)
        z = Y[:, sel].mean(1)
        gain, off = np.linalg.lstsq(np.c_[z, np.ones_like(z)], X[:, k], rcond=None)[0]
        D[:175, k][sel] = gain / sel.sum()
        D[175, k] = off
    candidates["boxcar_affine"] = D
    # (b) ridge with intercept (intercept not penalised)
    ym, xm = Y.mean(0), X.mean(0)
    Yc, Xc = Y - ym, X - xm
    W = np.linalg.solve(Yc.T @ Yc + args.alpha * len(Y) * np.eye(175), Yc.T @ Xc)
    candidates["ridge"] = np.vstack([W, xm - ym @ W])

    def apply(D, Y):
        return Y @ D[:175] + D[175]

    fits = {name: {"train_r2": r2_scores(X, apply(D, Y)), "val_r2": r2_scores(Xv, apply(D, Yv)),
                   "val_rmse_reflectance": [float(np.sqrt(((Xv[:, k] - apply(D, Yv)[:, k]) ** 2).mean())) for k in range(7)]}
            for name, D in candidates.items()}
    chosen = max(fits, key=lambda n: float(np.mean(fits[n]["val_r2"])))
    D = candidates[chosen]
    fit = {"chosen": chosen, "candidates": fits, "alpha": args.alpha, "n_fit_pixels": int(len(Y)),
           "n_val_pixels": int(len(Yv)), "fit_tiles": len(names), "train_split": SPLIT,
           "val_split": VAL_SPLIT, "shape": [176, 7], "last_row": "intercept"}
    print(json.dumps(fit, indent=2))

    Path("data").mkdir(exist_ok=True)
    np.save("data/degradation_175x7.npy", D.astype(np.float32))
    Path("data/degradation_fit.json").write_text(json.dumps(fit, indent=2))
    c = cfg.client()
    for f in ("degradation_175x7.npy", "degradation_fit.json"):
        c.upload_file(f"data/{f}", cfg.bucket, f"metadata/{f}")
    print("saved data/degradation_175x7.npy and uploaded to r2://metadata/")


if __name__ == "__main__":
    main()
