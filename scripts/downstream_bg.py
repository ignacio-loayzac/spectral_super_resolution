#!/usr/bin/env python
"""
Downstream experiments B (narrow-band indices) and G (absorption features) of
downstream_evaluation_protocol.md, on the OLI/Hyperion TEST split (scenes 11-15).

For every model the same pixel sample is used (128 valid pixels per test tile,
fixed seed per tile). Indices and features are computed on the real Hyperion
spectrum and on the reconstruction; agreement is measured with R² (1 - SS_res/SS_tot),
Pearson r, OLS slope and RMSE.

B · Indices (bands resolved by wavelength with spectral_sr.bands.band_at)
    PRI  = (R531 - R570) / (R531 + R570)                        vegetation (NDVI_real > 0.3)
    REP  = 700 + 40 * ((R670 + R780)/2 - R700) / (R740 - R700)  vegetation, Clevers linear model
    NDSI = (R1650 - R2200) / (R1650 + R2200)                    soil (NDVI_real < 0.2)
G · Absorption features, continuum = straight line between the window end bands
    chlorophyll 550-750 nm, leaf water 1100-1270 nm  (vegetation)
    Al-OH 2120-2250 nm, CO3 2270-2380 nm              (soil)
    band depth BD = 1 - min(R/Rc), area = ∫(1 - R/Rc) dλ, position of the minimum
    (parabolic refinement); position error and detection rate on pixels with BD_real > 0.02.

Usage:
    python scripts/downstream_bg.py --runs experiments/eval_h1_runs.json --out results/downstream_bg.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectral_sr.bands import band_at, band_window
from spectral_sr.datasets import SpectralDataset_npy
from spectral_sr.device import default_device
from spectral_sr.losses import valid_pixel_mask
from spectral_sr.storage import resolve_directory

import importlib.util
_spec = importlib.util.spec_from_file_location("evaluate_models", Path(__file__).with_name("evaluate_models.py"))
ev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ev)

PX_PER_TILE = 128
FEATURES = {"clorofila": (550, 750, "veg"), "agua_foliar": (1100, 1270, "veg"),
            "Al-OH": (2120, 2250, "soil"), "CO3": (2270, 2380, "soil")}


def sample_pixels(loader):
    """Fixed per-tile pixel sample (indices into H*W) among valid pixels."""
    picks = []
    for t, (x, y) in enumerate(_iter_tiles(loader)):
        m = valid_pixel_mask(y[None], x[None])[0].reshape(-1).numpy()
        idx = np.flatnonzero(m)
        rng = np.random.default_rng(t)
        picks.append(rng.choice(idx, min(PX_PER_TILE, len(idx)), replace=False) if len(idx) else idx)
    return picks


def _iter_tiles(loader):
    for xb, yb in loader:
        for i in range(xb.shape[0]):
            yield xb[i], yb[i]


def collect(model_fn, loader, picks):
    """Reflectance spectra (N, 175) at the sampled pixels; model_fn=None returns the ground truth."""
    out, t = [], 0
    with torch.no_grad():
        for xb, yb in loader:
            pb = yb if model_fn is None else model_fn(xb)
            pb = pb.float().cpu().numpy() / 1e4
            for i in range(pb.shape[0]):
                flat = pb[i].reshape(175, -1)
                out.append(flat[:, picks[t]].T)
                t += 1
    return np.concatenate(out)


def indices(S, ch):
    R = lambda k: S[:, ch[k]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (R(800) - R(670)) / (R(800) + R(670))
        pri = (R(531) - R(570)) / (R(531) + R(570))
        rep = 700 + 40 * ((R(670) + R(780)) / 2 - R(700)) / (R(740) - R(700))
        ndsi = (R(1650) - R(2200)) / (R(1650) + R(2200))
    return {"NDVI": ndvi, "PRI": pri, "REP": rep, "NDSI": ndsi}


def features(S, wl, idx):
    lam = wl[idx]
    R = S[:, idx]
    Rc = R[:, :1] + (R[:, -1:] - R[:, :1]) * (lam - lam[0]) / (lam[-1] - lam[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = R / Rc
    cr = np.where(np.isfinite(cr), cr, 1.0)
    j = np.argmin(cr, axis=1)
    bd = 1 - cr[np.arange(len(cr)), j]
    jj = np.clip(j, 1, len(lam) - 2)
    y0, y1, y2 = cr[np.arange(len(cr)), jj - 1], cr[np.arange(len(cr)), jj], cr[np.arange(len(cr)), jj + 1]
    denom = y0 - 2 * y1 + y2
    off = np.where(np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / denom, 0.0).clip(-1, 1)
    pos = lam[jj] + off * np.diff(lam).mean()
    area = np.trapz(1 - cr, lam, axis=1)
    return bd, area, pos


def agree(real, pred, sel):
    """Agreement on the selected pixels. R²/RMSE are outlier-sensitive (ratio indices explode when a
    denominator approaches 0), so robust metrics (Spearman, median absolute error) are reported too."""
    from scipy.stats import spearmanr
    ok = sel & np.isfinite(real) & np.isfinite(pred)
    r, p = real[ok], pred[ok]
    if len(r) < 10:
        return None
    ss_res, ss_tot = ((p - r) ** 2).sum(), ((r - r.mean()) ** 2).sum()
    slope = float(np.polyfit(r, p, 1)[0])
    return {"n": int(len(r)), "r2": float(1 - ss_res / ss_tot), "pearson": float(np.corrcoef(r, p)[0, 1]),
            "spearman": float(spearmanr(r, p).statistic), "slope": slope,
            "rmse": float(np.sqrt(((p - r) ** 2).mean())), "medae": float(np.median(np.abs(p - r)))}


MIN_REFL = 0.01  # physical validity: reflectance > 1% in every band an index/feature uses


def valid_bands(S, idx):
    return (S[:, idx] > MIN_REFL).all(axis=1)


def analyse(S_real, S_pred, wl, ch, masks):
    """Each metric is computed (a) on all masked pixels ("raw") and (b) only where BOTH the real and the
    predicted spectra are physically valid in the bands involved ("valid"); the share of pixels where the
    PREDICTION is invalid while the real spectrum is valid is reported as invalid_pred_rate."""
    ir, ip = indices(S_real, ch), indices(S_pred, ch)
    res = {"B": {}, "G": {}}
    bands_of = {"PRI": [ch[531], ch[570]], "REP": [ch[670], ch[700], ch[740], ch[780]], "NDSI": [ch[1650], ch[2200]]}
    cover_of = {"PRI": "veg", "REP": "veg", "NDSI": "soil"}
    for k in ("PRI", "REP", "NDSI"):
        sel = masks[cover_of[k]]
        if k == "REP":
            sel = sel & ((S_real[:, ch[740]] - S_real[:, ch[700]]) > 0.01) & (ir["REP"] > 680) & (ir["REP"] < 760)
        vr, vp = valid_bands(S_real, bands_of[k]), valid_bands(S_pred, bands_of[k])
        res["B"][k] = {"raw": agree(ir[k], ip[k], sel), "valid": agree(ir[k], ip[k], sel & vr & vp),
                       "invalid_pred_rate": float((~vp[sel & vr]).mean())}
    for name, (a, b, cover) in FEATURES.items():
        idx = band_window(wl, a, b)
        bdr, ar, pr = features(S_real, wl, idx)
        bdp, ap, pp = features(S_pred, wl, idx)
        vr, vp = valid_bands(S_real, idx), valid_bands(S_pred, idx)
        sel = masks[cover] & vr & vp
        present = sel & (bdr > 0.02)
        d_lambda = np.abs(pp - pr)[present]
        res["G"][name] = {"bd": agree(bdr, bdp, sel), "area": agree(ar, ap, sel),
                          "bd_raw": agree(bdr, bdp, masks[cover]),
                          "invalid_pred_rate": float((~vp[masks[cover] & vr]).mean()),
                          "pos_err_median_nm": float(np.median(d_lambda)) if present.any() else None,
                          "pos_err_p90_nm": float(np.percentile(d_lambda, 90)) if present.any() else None,
                          "detection_rate": float((bdp[present] > 0.02).mean()) if present.any() else None,
                          "n_present": int(present.sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/eval_h1_runs.json")
    ap.add_argument("--extra", default=None, help="optional JSON with more specs (e.g. 2020 checkpoints)")
    ap.add_argument("--out", default="results/downstream_bg.json")
    args = ap.parse_args()
    device = torch.device(default_device())
    wl = np.load("data/wavelengths_175.npy")
    ch = {lam: band_at(wl, lam) for lam in (531, 570, 670, 700, 740, 780, 800, 1650, 2200)}
    ds = SpectralDataset_npy(resolve_directory("landsat_test_npy"), resolve_directory("hyperion_test_npy"))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    picks = sample_pixels(loader)
    S_real = collect(None, loader, picks)
    Path("outputs/downstream_cache").mkdir(parents=True, exist_ok=True)
    np.save("outputs/downstream_cache/_real.npy", S_real.astype(np.float32))
    ndvi = indices(S_real, ch)["NDVI"]
    masks = {"veg": ndvi > 0.3, "soil": ndvi < 0.2}
    print(f"pixels: {len(S_real):,} | vegetación {masks['veg'].mean():.1%} | suelo {masks['soil'].mean():.1%}")

    specs = json.loads(Path(args.runs).read_text())
    if args.extra:
        specs += json.loads(Path(args.extra).read_text())
    per_model = {}
    for spec in specs:
        s = ev.resolve_spec(spec)
        model = ev._load(s, device)
        scale = float(s["data_scale"])
        fn = lambda xb, m=model, sc=scale: (m(xb.to(device) / sc).reshape(xb.shape[0], 175, *xb.shape[2:]) * sc)
        S_pred = collect(fn, loader, picks)
        cache = Path("outputs/downstream_cache"); cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / f"{spec['name'].replace(' ', '_')}.npy", S_pred.astype(np.float32))
        per_model[spec["name"]] = analyse(S_real, S_pred, wl, ch, masks)
        b = per_model[spec["name"]]["B"]
        print(f"{spec['name']:<26} R² válido PRI {b['PRI']['valid']['r2']:+.3f} | REP {b['REP']['valid']['r2']:+.3f} | "
              f"NDSI {b['NDSI']['valid']['r2']:+.3f} | inválidos PRI {b['PRI']['invalid_pred_rate']:.1%}", flush=True)
    Path(args.out).write_text(json.dumps({"pixels": int(len(S_real)), "mask_share": {k: float(v.mean()) for k, v in masks.items()},
                                          "per_model": per_model}, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
