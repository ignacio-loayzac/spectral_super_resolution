#!/usr/bin/env python
"""Do reconstructed spectra predict chlorophyll absorption / red-edge position better than OLI itself?

On the downstream test sample (128 valid pixels per test tile, same picks as scripts/downstream_bg.py),
vegetation pixels (NDVI from the real Hyperion spectrum > 0.3), physically valid spectra:
  targets   chlorophyll band depth (550-750 nm, continuum removed) and REP (Clevers), from REAL Hyperion
  predictors
    - the same product computed on each model's reconstruction (outputs/downstream_cache/*.npy)
    - broadband OLI indices: NDVI (B5,B4), GNDVI (B5,B3), simple ratio B5/B4, red-edge-free proxies
    - direct ridge regression OLI 7 bands (+ the three indices) -> target, fitted on vegetation pixels
      of the 8 QC training scenes (same pixel sampling rule), evaluated on the test sample
Metrics (calibration-free): Spearman rho and r² = squared Pearson correlation (the R² of the best
linear calibration); for predictors already in target units, also R² against the identity.
Output: results/broadband_vs_reconstruction.json / .md
"""
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from spectral_sr.bands import band_at, band_window
from spectral_sr.datasets import SpectralDataset_npy
from spectral_sr.losses import valid_pixel_mask
from spectral_sr.storage import resolve_directory

_s = importlib.util.spec_from_file_location("ds", Path(__file__).with_name("downstream_bg.py"))
ds = importlib.util.module_from_spec(_s); _s.loader.exec_module(ds)
TRAIN_SCENES = [1, 2, 3, 4, 7, 8, 9, 10]


def oli_at(loader, picks):
    out, t = [], 0
    for xb, _ in loader:
        xb = xb.float().numpy() / 1e4
        for i in range(xb.shape[0]):
            out.append(xb[i].reshape(7, -1)[:, picks[t]].T); t += 1
    return np.concatenate(out)


def oli_features(X):
    B3, B4, B5 = X[:, 2], X[:, 3], X[:, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"NDVI (OLI)": (B5 - B4) / (B5 + B4), "GNDVI (OLI)": (B5 - B3) / (B5 + B3), "SR B5/B4 (OLI)": B5 / B4}


def targets(S, wl, ch):
    idx = band_window(wl, 550, 750)
    bd, _, _ = ds.features(S, wl, idx)
    rep = ds.indices(S, ch)["REP"]
    return {"BD clorofila": bd, "REP": rep}, idx


def design(X):
    f = oli_features(X)
    return np.c_[X, f["NDVI (OLI)"], f["GNDVI (OLI)"], f["SR B5/B4 (OLI)"], np.ones(len(X))]


def ridge(A, y, alpha=1e-3):
    P = np.eye(A.shape[1]) * alpha * len(A); P[-1, -1] = 0
    return np.linalg.solve(A.T @ A + P, A.T @ y)


def metrics(pred, real, identity=False):
    ok = np.isfinite(pred) & np.isfinite(real)
    p, r = pred[ok], real[ok]
    out = {"n": int(ok.sum()), "rho": float(spearmanr(p, r).statistic), "r2_corr": float(np.corrcoef(p, r)[0, 1] ** 2)}
    if identity:
        out["R2_identity"] = float(1 - ((p - r) ** 2).sum() / ((r - r.mean()) ** 2).sum())
    return out


def sample(dirs_multi, dirs_hyp, scenes=None):
    d = SpectralDataset_npy(dirs_multi, dirs_hyp, scenes=scenes) if scenes else SpectralDataset_npy(dirs_multi, dirs_hyp)
    loader = DataLoader(d, batch_size=32, shuffle=False, num_workers=0)
    picks = ds.sample_pixels(loader)
    return ds.collect(None, loader, picks), oli_at(loader, picks)


def main():
    wl = np.load("data/wavelengths_175.npy")
    ch = {lam: band_at(wl, lam) for lam in (531, 570, 670, 700, 740, 780, 800, 1650, 2200)}
    # test sample (identical picks to downstream_bg.py)
    S_te, X_te = sample(resolve_directory("landsat_test_npy"), resolve_directory("hyperion_test_npy"))
    assert np.allclose(S_te, np.load("outputs/downstream_cache/_real.npy"), atol=1e-4)
    S_tr, X_tr = sample(resolve_directory("landsat_train_npy"), resolve_directory("hyperion_train_npy"), TRAIN_SCENES)
    res = {}
    for split, S, X in (("test", S_te, X_te), ("train", S_tr, X_tr)):
        T, idx = targets(S, wl, ch)
        veg = ds.indices(S, ch)["NDVI"] > 0.3
        sel = {"BD clorofila": veg & ds.valid_bands(S, idx),
               "REP": veg & ds.valid_bands(S, [ch[670], ch[700], ch[740], ch[780]])
                      & ((S[:, ch[740]] - S[:, ch[700]]) > 0.01) & (T["REP"] > 680) & (T["REP"] < 760)}
        res[split] = (T, sel, X)
    T_te, sel_te, _ = res["test"]; T_tr, sel_tr, _ = res["train"]
    out = {}
    for tgt in ("BD clorofila", "REP"):
        st, sr = sel_te[tgt], sel_tr[tgt]
        rows = {}
        for name, f in oli_features(X_te).items():
            rows[name] = metrics(f[st], T_te[tgt][st])
        A_tr = design(X_tr[sr]); ok = np.isfinite(A_tr).all(1) & np.isfinite(T_tr[tgt][sr])
        w = ridge(A_tr[ok], T_tr[tgt][sr][ok])
        rows["Regresión directa OLI → producto"] = metrics(design(X_te[st]) @ w, T_te[tgt][st], identity=True)
        groups = defaultdict(list)
        for f in sorted(Path("outputs/downstream_cache").glob("*.npy")):
            if f.name.startswith("_"):
                continue
            arm = f.stem.split("_")[0] if "seed" in f.stem else f.stem
            P = np.load(f).astype(np.float64)
            Tp, _ = targets(P, wl, ch)
            valid_p = ds.valid_bands(P, band_window(wl, 550, 750)) if tgt == "BD clorofila" else np.ones(len(P), bool)
            m = st & valid_p
            groups[arm].append(metrics(Tp[tgt][m], T_te[tgt][m], identity=True))
        for arm, ms in groups.items():
            rows[f"Reconstrucción {arm}"] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        out[tgt] = rows
    Path("results/broadband_vs_reconstruction.json").write_text(json.dumps(out, indent=1))
    md = []
    for tgt, rows in out.items():
        md += [f"## {tgt} (prueba, vegetación)", "", "| Predictor | n | ρ | r² (calibración lineal) | R² (identidad) |", "| :--- | ---: | ---: | ---: | ---: |"]
        for name, m in sorted(rows.items(), key=lambda kv: -kv[1]["rho"]):
            md.append(f"| {name} | {m['n']} | {m['rho']:.3f} | {m['r2_corr']:.3f} | {m.get('R2_identity', float('nan')):.3f} |")
        md.append("")
    Path("results/broadband_vs_reconstruction.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
