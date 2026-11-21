#!/usr/bin/env python
"""
Generate the RunPod job plans for BACKLOG E-M (phase 1: validation sweeps; phase 2: final runs).

Coefficients are set RELATIVE to the content (MSE) term, measured on real data with a
trained 2020 prediction (content MSE ~ 2.9e-3 in reflectance^2):
  raw Gram (2020 AE)          ~ 1.2e-6  -> beta_raw  = r * 2.9e-3 / 1.2e-6
  correlation Gram (AE)       ~ G_corr  -> beta_corr = r * 2.9e-3 / G_corr
  feature term (AE)           ~ F       -> gamma     = r * 2.9e-3 / F
  consistency (bands 2,4,5,6) ~ 9.1e-4  -> lambda    = r * 2.9e-3 / 9.1e-4
with r in {0.1, 1, 10} (perceptual term = 10%, 100%, 1000% of the content term).
Magnitudes of the spectral AE are passed on the command line once it is trained.

Usage: python experiments/make_plans.py --gcorr-spec 0.05 --feat-spec 0.01
"""

import argparse
import json
from pathlib import Path

CONTENT = 2.9e-3
COMMON = {
    "mode": "sr", "model": "hourglass", "encoder_channels": [150, 100, 75],
    "hyperion_train_dir": "r2://splits/v2/hyperion_train_npy", "landsat_train_dir": "r2://splits/v2/landsat_train_npy",
    "hyperion_val_dir": "r2://splits/v1/hyperion_val_npy", "landsat_val_dir": "r2://splits/v1/landsat_val_npy",
    "data_scale": 10000.0, "augment": True, "use_mask": True, "lr": 1e-3, "batch_size": 32, "num_workers": 12,
    "lr_scheduler": "plateau", "checkpoint_every": 0, "content_loss_coeff": 1.0,
}
AE2020 = {"loss": "perceptual", "ae_checkpoint": "r2://Models/AEHG_150_100_75.pt", "ae_kernel_mode": "spatial",
          "ae_input_scale": 10000.0, "gram_mode": "raw"}
AESPEC = {"loss": "perceptual", "ae_checkpoint": "r2://Models/AE_spectral_150_100_75_v2.pt",
          "ae_kernel_mode": "spectral", "ae_input_scale": 1.0, "gram_mode": "correlation"}
CONS = {"consistency_coeff": None, "degradation_path": "r2://metadata/degradation_175x7.npy",
        "consistency_bands": [2, 4, 5, 6]}


def coef(r, magnitude):
    return float(f"{r * CONTENT / magnitude:.3g}")


def job(name, seed, epochs, patience, **kw):
    body = {**COMMON, **kw, "seed": seed, "epochs": epochs, "early_stopping_patience": patience,
            "experiment_name": name}
    body.setdefault("loss", "mse")
    body.setdefault("style_loss_coeff", 0.0)
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcorr-spec", type=float, required=True, help="correlation-Gram magnitude with the spectral AE")
    ap.add_argument("--feat-spec", type=float, required=True, help="feature-term magnitude with the spectral AE")
    args = ap.parse_args()
    R = (0.1, 1.0, 10.0)
    sweep = dict(seed=42, epochs=20, patience=5)
    p1 = [job("sweep/M0_mse", **sweep)]
    for r in R:
        p1.append(job(f"sweep/M1_raw_r{r:g}", **sweep, **AE2020, style_loss_coeff=coef(r, 1.2e-6)))
    for r in R:
        for rg in (0.0, 1.0):
            p1.append(job(f"sweep/M2_corr_r{r:g}_feat{rg:g}", **sweep, **AESPEC, style_loss_coeff=coef(r, args.gcorr_spec),
                          feature_loss_coeff=coef(rg, args.feat_spec) if rg else 0.0))
    for rl in (0.1, 1.0):
        p1.append(job(f"sweep/M3_cons_r{rl:g}", **sweep, **AESPEC, style_loss_coeff=coef(1.0, args.gcorr_spec),
                      **{**CONS, "consistency_coeff": coef(rl, 9.1e-4)}))
    Path("experiments/phase1_sweep.json").write_text(json.dumps(p1, indent=2))
    print(f"phase1: {len(p1)} jobs -> experiments/phase1_sweep.json")
    for j in p1:
        print(f"  {j['experiment_name']:<32} beta={j.get('style_loss_coeff', 0)} gamma={j.get('feature_loss_coeff', 0)} "
              f"lambda={j.get('consistency_coeff', 0)}")


if __name__ == "__main__":
    main()
