#!/usr/bin/env python
"""Per-pixel linear baseline OLI (7 bands) -> Hyperion (175 bands), ridge regression with intercept.

  ŷ_p = x_p W + b,   (W, b) = argmin Σ_p ||x_p W + b - y_p||² + α·n·||W||_F²   (intercept not penalised)

Trained on the valid pixels of the training tiles of the 8 QC-passing scenes (1-4, 7-10; the
same data as the final Study 3 models), in reflectance. α is chosen with the same scene
cross-validation as Study 3 (validation groups {1,2}, {3,4}, {7,8}, {9,10}), minimising the
pooled validation RMSE. Only sufficient statistics are accumulated (XᵀX, XᵀY, ΣY², n per scene),
so the fit is exact and cheap. The model is saved as a 1x1 convolution state dict so that
scripts/evaluate_models.py and scripts/downstream_bg.py can evaluate it ("model": "linear").

Usage: python scripts/linear_baseline.py --out results/linear_baseline.pt
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from spectral_sr.storage import resolve_directory

SCENES = [1, 2, 3, 4, 7, 8, 9, 10]
FOLDS = [[1, 2], [3, 4], [7, 8], [9, 10]]
ALPHAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def stats_by_scene():
    hd, ld = Path(resolve_directory("hyperion_train_npy")), Path(resolve_directory("landsat_train_npy"))
    S = defaultdict(lambda: {"XtX": np.zeros((8, 8)), "XtY": np.zeros((8, 175)), "YtY": 0.0, "n": 0})
    for f in sorted(hd.glob("*.npy")):
        sc = int(f.name.split("_")[0])
        if sc not in SCENES:
            continue
        y = np.load(f).reshape(-1, 175).astype(np.float64) / 1e4
        x = np.load(ld / f.name).reshape(-1, 7).astype(np.float64) / 1e4
        v = (np.abs(y).sum(1) > 1e-6) & (x != 0).all(1)
        xa = np.c_[x[v], np.ones(v.sum())]
        s = S[sc]
        s["XtX"] += xa.T @ xa; s["XtY"] += xa.T @ y[v]; s["YtY"] += float((y[v] ** 2).sum()); s["n"] += int(v.sum())
    return S


def fit(st, alpha):
    XtX, XtY, n = st["XtX"], st["XtY"], st["n"]
    P = np.eye(8) * alpha * n; P[7, 7] = 0.0          # do not penalise the intercept
    return np.linalg.solve(XtX + P, XtY)               # (8, 175): rows 0-6 = W, row 7 = b


def sse(st, Wb):
    return st["YtY"] - 2 * np.sum(Wb * st["XtY"]) + np.sum(Wb * (st["XtX"] @ Wb))


def merge(S, scenes):
    out = {"XtX": 0, "XtY": 0, "YtY": 0.0, "n": 0}
    for sc in scenes:
        for k in out:
            out[k] = out[k] + S[sc][k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/linear_baseline.pt")
    args = ap.parse_args()
    S = stats_by_scene()
    cv = {}
    for a in ALPHAS:
        e = n = 0.0
        fold_rmse = []
        for val in FOLDS:
            Wb = fit(merge(S, [s for s in SCENES if s not in val]), a)
            vs = merge(S, val)
            fe = sse(vs, Wb); e += fe; n += vs["n"] * 175
            fold_rmse.append(math.sqrt(fe / (vs["n"] * 175)))
        cv[a] = {"pooled_rmse": math.sqrt(e / n), "fold_rmse": fold_rmse}
        print(f"alpha {a:g}: CV RMSE {cv[a]['pooled_rmse']:.5f}  folds {[round(r, 4) for r in fold_rmse]}")
    best = min(cv, key=lambda a: cv[a]["pooled_rmse"])
    Wb = fit(merge(S, SCENES), best)
    conv = torch.nn.Conv2d(7, 175, 1)
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(Wb[:7].T.copy()).float().view(175, 7, 1, 1))
        conv.bias.copy_(torch.from_numpy(Wb[7].copy()).float())
    torch.save(conv.state_dict(), args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"alpha": best, "train_scenes": SCENES, "cv_folds": FOLDS, "n_train_pixels": merge(S, SCENES)["n"],
         "cv": {str(a): v for a, v in cv.items()}}, indent=1))
    print(f"chosen alpha {best:g}; saved {args.out}")


if __name__ == "__main__":
    main()
