#!/usr/bin/env python
"""
Plans for the preregistered H1/H5/H2 study (experiments/PREREGISTRO_H1.md).

  python experiments/make_plans_h1.py cv
      -> experiments/phase3_cv.json: 5 arms x 4 scene folds, 1 seed, <=30 epochs, patience 6
  python experiments/make_plans_h1.py final results/runpod_phase3_cv_<ts>.json
      -> experiments/phase4_final.json: 5 arms x seeds 1-3, trained on the 8 QC scenes for
         E = median best epoch of the arm's 4 folds (no early stopping)
"""

import json
import statistics
import sys
from pathlib import Path

QC_SCENES = [1, 2, 3, 4, 7, 8, 9, 10]
FOLDS = [[1, 2], [3, 4], [7, 8], [9, 10]]
AE_PATH = "r2://Models/AE_bottleneck16_scenes1-10.pt"
PCA_PATH = "r2://Models/pca16_scenes1-10.npz"
COMMON = {
    "mode": "sr", "encoder_channels": [150, 100, 75], "kernel_mode": "spatial",
    "hyperion_train_dir": "r2://hyperion_train_npy", "landsat_train_dir": "r2://landsat_train_npy",
    "hyperion_val_dir": "r2://hyperion_train_npy", "landsat_val_dir": "r2://landsat_train_npy",
    "data_scale": 10000.0, "augment": True, "use_mask": True, "lr": 1e-3, "batch_size": 32, "num_workers": 12,
    "lr_scheduler": "plateau", "checkpoint_every": 0, "loss": "mse", "content_loss_coeff": 1.0,
    "style_loss_coeff": 0.0,
}
CONS = {"consistency_coeff": 0.319, "degradation_path": "r2://metadata/degradation_175x7.npy",
        "consistency_bands": [2, 4, 5, 6]}
LATENT_AE = {"model": "latent", "latent_dim": 16, "decoder_type": "ae", "decoder_path": AE_PATH}
LATENT_PCA = {"model": "latent", "latent_dim": 16, "decoder_type": "pca", "decoder_path": PCA_PATH}
ARMS = {
    "M0": {"model": "hourglass"},
    "C": {"model": "hourglass", **CONS},
    "P": {**LATENT_PCA},
    "A": {**LATENT_AE},
    "AC": {**LATENT_AE, **CONS},
}


def cv():
    plan = []
    for arm, cfg in ARMS.items():
        for i, fold in enumerate(FOLDS):
            plan.append({**COMMON, **cfg, "seed": 42, "epochs": 30, "early_stopping_patience": 6,
                         "train_scenes": [s for s in QC_SCENES if s not in fold], "val_scenes": fold,
                         "experiment_name": f"h1cv/{arm}/fold{i}"})
    Path("experiments/phase3_cv.json").write_text(json.dumps(plan, indent=2))
    print(f"{len(plan)} CV jobs -> experiments/phase3_cv.json")


def final(record_path):
    rec = json.loads(Path(record_path).read_text())
    best = {}
    for j in rec["jobs"]:
        arm = j["experiment_name"].split("/")[1]
        if j.get("status") == "done" and j.get("best"):
            best.setdefault(arm, []).append(j["best"]["epoch"])
    summary, plan = {}, []
    for arm, cfg in ARMS.items():
        eps = best.get(arm, [])
        if len(eps) != len(FOLDS):
            raise SystemExit(f"arm {arm}: {len(eps)} finished folds, expected {len(FOLDS)}")
        E = int(round(statistics.median(eps)))
        summary[arm] = {"best_epochs": eps, "E": E}
        for seed in (1, 2, 3):
            plan.append({**COMMON, **cfg, "seed": seed, "epochs": E, "early_stopping_patience": 0,
                         "train_scenes": QC_SCENES, "val_scenes": [2],
                         "experiment_name": f"h1final/{arm}/seed{seed}"})
    Path("experiments/phase4_final.json").write_text(json.dumps(plan, indent=2))
    Path("experiments/phase4_epochs.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"{len(plan)} final jobs -> experiments/phase4_final.json")


if __name__ == "__main__":
    cv() if sys.argv[1] == "cv" else final(sys.argv[2])
