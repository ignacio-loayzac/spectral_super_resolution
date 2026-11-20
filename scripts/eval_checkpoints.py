#!/usr/bin/env python
"""
Evaluate the 2020 checkpoints on the OLI/Hyperion test split (BACKLOG T-013).

Reproduces the thesis table "Experimental results for the OLI/Hyperion dataset"
(MSE, RMSE, RMSE scaled to reflectance, SSIM, SAM). Because the exact
aggregation used in 2020 is not documented, every metric is reported both as
the mean of per-batch values (as the legacy training loop logged them) and as
a global value over all pixels.

Usage:
    python scripts/eval_checkpoints.py [--batch-size 32] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectral_sr.datasets import SpectralDataset_npy
from spectral_sr.device import default_device
from spectral_sr.metrics import SAM, SSIM
from spectral_sr.models import Hourglass, Koundinya2D_CNN
from spectral_sr.storage import resolve_directory, resolve_path

CHECKPOINTS = {
    "UNet (hourglass) + MSE": ("hourglass", "Models/hg_SR_150_100_75.pt"),
    "UNet (hourglass) + Perceptual": ("hourglass", "Models/hg_SR_150_100_75_perceptual.pt"),
    "CNN (Koundinya) + MSE": ("koundinya", "Models/koundinya_mse.pt"),
    "CNN (Koundinya) + Perceptual": ("koundinya", "Models/koundinya_perceptual.pt"),
}
THESIS = {  # Masters_Thesis/Results.tex, OLI/Hyperion table: MSE, RMSE, RMSE refl., SSIM, SAM
    "UNet (hourglass) + MSE": (397764.405, 630.685, 0.06306, 0.49794, 0.19773),
    "UNet (hourglass) + Perceptual": (353297.060, 594.387, 0.05943, 0.50520, 0.19094),
    "CNN (Koundinya) + MSE": (537094.030, 732.866, 0.07328, 0.36121, 0.23311),
    "CNN (Koundinya) + Perceptual": (461736.502, 679.511, 0.06795, 0.42282, 0.21195),
}


def build(kind: str) -> torch.nn.Module:
    if kind == "hourglass":
        return Hourglass(in_channels=7, out_channels=175, ub_out_channels=150, mb_out_channels=100, lb_out_channels=75)
    return Koundinya2D_CNN()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/t013_eval_2020_checkpoints.json")
    args = ap.parse_args()
    device = torch.device(args.device or default_device())

    hyp = resolve_directory("hyperion_test_npy", progress=lambda d, t, b: print(f"\r hyperion_test {d}/{t}", end=""))
    ls = resolve_directory("landsat_test_npy")
    print()
    ds = SpectralDataset_npy(ls, hyp)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    ssim, sam = SSIM(), SAM()
    results = {}
    for name, (kind, key) in CHECKPOINTS.items():
        model = build(kind)
        missing = model.load_state_dict(torch.load(resolve_path(f"r2://{key}"), map_location="cpu"), strict=True)
        model.to(device).eval()
        b_mse, b_ssim, b_sam = [], [], []
        se_sum, n_el, ang_sum, n_px = 0.0, 0, 0.0, 0
        t0 = time.perf_counter()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                p = model(x).reshape(y.shape)
                err = (p - y).float().cpu().double()
                b_mse.append(float((err ** 2).mean()))
                b_ssim.append(float(ssim(p, y)))
                b_sam.append(float(sam(p, y)))  # degrees (mean over pixels of the batch)
                se_sum += float((err ** 2).sum()); n_el += err.numel()
                cos = (p * y).sum(1) / (p.norm(dim=1) * y.norm(dim=1) + 1e-8)
                ang_sum += float(torch.arccos(cos.clamp(-1, 1)).cpu().double().sum()); n_px += cos.numel()
        mse_b = float(np.mean(b_mse)); mse_g = se_sum / n_el
        r = {
            "mse_batch_mean": mse_b, "mse_global": mse_g,
            "rmse_of_batch_mean_mse": math.sqrt(mse_b), "rmse_batch_mean": float(np.mean(np.sqrt(b_mse))),
            "rmse_global": math.sqrt(mse_g), "rmse_reflectance": math.sqrt(mse_g) / 10000,
            "ssim_batch_mean": float(np.mean(b_ssim)),
            "sam_deg_batch_mean": float(np.mean(b_sam)), "sam_rad_global": ang_sum / n_px,
            "seconds": round(time.perf_counter() - t0, 1),
            "thesis": dict(zip(["mse", "rmse", "rmse_reflectance", "ssim", "sam"], THESIS[name])),
        }
        results[name] = r
        t = r["thesis"]
        print(f"\n{name}  ({r['seconds']} s on {device})")
        print(f"  MSE   batch-mean {mse_b:12.1f} | global {mse_g:12.1f} | tesis {t['mse']:12.1f}  ({100*(mse_b/t['mse']-1):+.1f}%)")
        print(f"  RMSE  sqrt(mean) {math.sqrt(mse_b):8.2f} | mean(sqrt) {r['rmse_batch_mean']:8.2f} | tesis {t['rmse']:8.2f}")
        print(f"  SSIM  {r['ssim_batch_mean']:.5f} | tesis {t['ssim']:.5f}")
        print(f"  SAM   {r['sam_rad_global']:.5f} rad ({r['sam_deg_batch_mean']:.3f}°) | tesis {t['sam']:.5f}")
    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({"device": str(device), "n_test_tiles": len(ds), "results": results}, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
