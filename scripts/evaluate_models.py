#!/usr/bin/env python
"""
Reproducible test-set evaluation of spectral SR models (BACKLOG T-013 / T-019).

Every model is evaluated on the full OLI/Hyperion test split (scenes 11-15) with
the same masked metrics, in reflectance units:

  RMSE, MAE            over valid pixels and all 175 bands
  SAM (rad)            mean spectral angle over valid pixels
  SSIM                 legacy SSIM (as in the thesis), unmasked, per tile
  RMSE per band group  OLI-like groups (CA, Blue, Green, Red, NIR, SWIR1, SWIR2)
                       using ALL channels of the group (fixes legacy get_metrics)
  95% CI               percentile bootstrap over test tiles (2 000 resamples)

Model specs come from --runs (JSON list) or default to the four 2020 checkpoints.
A spec is {"name", "checkpoint", "model": "hourglass"|"koundinya",
"kernel_mode", "data_scale", "encoder_channels"}; a spec may instead give
"run": "<r2 experiments prefix>" (experiment_name/timestamp) whose model_meta.json and checkpoint_epoch_best.pt
are fetched from the experiments bucket.

Usage:
    python scripts/evaluate_models.py --out results/eval_2020.json
    python scripts/evaluate_models.py --runs runs.json --out results/eval_ablation.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectral_sr.datasets import SpectralDataset_npy
from spectral_sr.device import default_device
from spectral_sr.losses import spectral_angle, valid_pixel_mask
from spectral_sr.metrics import SSIM
from spectral_sr.models import Hourglass, Koundinya2D_CNN
from spectral_sr.storage import R2Config, resolve_directory, resolve_path

DEFAULT_RUNS = [
    {"name": "2020 UNet + MSE", "checkpoint": "r2://Models/hg_SR_150_100_75.pt", "model": "hourglass"},
    {"name": "2020 UNet + Perceptual", "checkpoint": "r2://Models/hg_SR_150_100_75_perceptual.pt", "model": "hourglass"},
    {"name": "2020 CNN + MSE", "checkpoint": "r2://Models/koundinya_mse.pt", "model": "koundinya"},
    {"name": "2020 CNN + Perceptual", "checkpoint": "r2://Models/koundinya_perceptual.pt", "model": "koundinya"},
]
GROUPS_NM = {"CA": (430, 450), "Blue": (450, 510), "Green": (530, 590), "Red": (640, 670),
             "NIR": (850, 880), "SWIR1": (1570, 1650), "SWIR2": (2110, 2290)}


def resolve_spec(spec: dict) -> dict:
    spec = dict(spec)
    if "run" in spec:
        exp = R2Config.from_env().for_experiments()
        prefix = spec["run"].rstrip("/")
        meta = json.loads(Path(resolve_path(f"r2://{prefix}/model_meta.json", exp)).read_text())
        spec.setdefault("checkpoint", f"r2://{prefix}/checkpoint_epoch_best.pt")
        spec.setdefault("model", meta["model"]["model_type"])
        spec.setdefault("kernel_mode", meta["model"].get("kernel_mode", "spatial"))
        spec.setdefault("encoder_channels", meta["model"].get("encoder_channels", [150, 100, 75]))
        spec.setdefault("data_scale", meta.get("data_scale", 1.0))
        spec["_bucket"] = exp
    spec.setdefault("kernel_mode", "spatial")
    spec.setdefault("encoder_channels", [150, 100, 75])
    spec.setdefault("data_scale", 1.0)
    return spec


def build(spec: dict) -> torch.nn.Module:
    if spec["model"] == "koundinya":
        return Koundinya2D_CNN()
    ec = spec["encoder_channels"]
    return Hourglass(7, 175, ec[0], ec[1], ec[2], kernel_mode=spec["kernel_mode"])


def evaluate(spec, loader, device, groups, n_boot=2000):
    model = build(spec)
    ckpt = resolve_path(spec["checkpoint"], spec.get("_bucket"))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()
    scale = float(spec["data_scale"])
    ssim = SSIM()
    per_tile = {"se": [], "ae": [], "ang": [], "npx": [], "ssim": []}
    grp_se = {g: 0.0 for g in groups}
    n_total = 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:  # loader yields raw stored values (x1e4)
            x, y = x.to(device), y.to(device)
            p = model(x / scale).reshape(y.shape) * scale
            m = valid_pixel_mask(y, x)
            pr, yr = p.float().cpu().double() / 1e4, y.float().cpu().double() / 1e4  # reflectance
            mc = m.cpu()
            err = pr - yr
            se = (err ** 2).mean(1)
            ae = err.abs().mean(1)
            ang = spectral_angle(pr, yr)
            for i in range(x.shape[0]):
                mi = mc[i]
                n = float(mi.sum())
                per_tile["npx"].append(n)
                per_tile["se"].append(float(se[i][mi].sum()))
                per_tile["ae"].append(float(ae[i][mi].sum()))
                per_tile["ang"].append(float(ang[i][mi].sum()))
                per_tile["ssim"].append(float(ssim(p[i:i + 1], y[i:i + 1])))
            for g, idx in groups.items():
                grp_se[g] += float((err[:, idx] ** 2).mean(1)[mc].sum())
            n_total += float(mc.sum())
    T = {k: np.array(v) for k, v in per_tile.items()}

    def stats(sel):
        n = T["npx"][sel].sum()
        return {"rmse": math.sqrt(T["se"][sel].sum() / n), "mae": T["ae"][sel].sum() / n,
                "sam": T["ang"][sel].sum() / n, "ssim": float(T["ssim"][sel].mean())}

    point = stats(np.arange(len(T["npx"])))
    rng = np.random.default_rng(0)
    boots = [stats(rng.integers(0, len(T["npx"]), len(T["npx"]))) for _ in range(n_boot)]
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))]
          for k in point}
    return {"name": spec["name"], "checkpoint": spec["checkpoint"], "kernel_mode": spec["kernel_mode"],
            "data_scale": scale, **point, "ci95": ci,
            "rmse_by_group": {g: math.sqrt(v / n_total) for g, v in grp_se.items()},
            "valid_pixel_fraction": float(T["npx"].sum() / (len(T["npx"]) * 64 * 64)),
            "per_tile": {"rmse": np.sqrt(T["se"] / np.maximum(T["npx"], 1)).round(6).tolist(),
                         "sam": (T["ang"] / np.maximum(T["npx"], 1)).round(6).tolist(),
                         "npx": T["npx"].astype(int).tolist()},
            "seconds": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="JSON file with a list of model specs")
    ap.add_argument("--out", default="results/eval_2020.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    device = torch.device(args.device or default_device())
    specs = json.loads(Path(args.runs).read_text()) if args.runs else DEFAULT_RUNS

    wl = np.load("data/wavelengths_175.npy")
    groups = {g: np.where((wl >= a) & (wl <= b))[0] for g, (a, b) in GROUPS_NM.items()}
    ds = SpectralDataset_npy(resolve_directory("landsat_test_npy"), resolve_directory("hyperion_test_npy"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []
    for spec in specs:
        r = evaluate(resolve_spec(spec), loader, device, groups)
        results.append(r)
        print(f"{r['name']:<40} RMSE {r['rmse']:.5f} [{r['ci95']['rmse'][0]:.5f}, {r['ci95']['rmse'][1]:.5f}] | "
              f"SAM {r['sam']:.4f} rad | SSIM {r['ssim']:.4f} | MAE {r['mae']:.5f} ({r['seconds']} s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "n_test_tiles": len(ds), "metrics_units": "reflectance",
                               "masked": True, "results": results}, indent=2))
    md = ["| Modelo | RMSE (refl.) [IC 95%] | SAM (rad) [IC 95%] | SSIM | MAE |", "| :--- | :---: | :---: | :---: | :---: |"]
    for r in results:
        c = r["ci95"]
        md.append(f"| {r['name']} | {r['rmse']:.4f} [{c['rmse'][0]:.4f}, {c['rmse'][1]:.4f}] | "
                  f"{r['sam']:.4f} [{c['sam'][0]:.4f}, {c['sam'][1]:.4f}] | {r['ssim']:.3f} | {r['mae']:.4f} |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"saved {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
