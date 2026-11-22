#!/usr/bin/env python
"""
Plans for the follow-up study (not preregistered): residual models over the linear baseline and
networks restricted to 1x1 convolutions. Same protocol as Study 3 (experiments/make_plans_h1.py).

  python experiments/make_plans_reg.py cv
      -> experiments/phase6_cv.json: 5 arms x 4 scene folds, seed 42, <=30 epochs, patience 6
         (residual arms use a linear baseline fitted WITHOUT the fold's validation scenes)
  python experiments/make_plans_reg.py final results/runpod_phase6_cv_<ts>.json
      -> experiments/phase6_final.json: 5 arms x seeds 1-3 on the 8 QC scenes, E = median best epoch
Arms:
  LR   linear baseline + gated Hourglass correction (spatial)
  LRC  LR + spectral consistency
  LR1  LR with a 1x1 (spectral) Hourglass
  M0s  Hourglass with 1x1 convolutions, direct 175 bands
  ACs  AE-16 prior + consistency with a 1x1 generator
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("h1", Path(__file__).with_name("make_plans_h1.py"))
h1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(h1)
QC_SCENES, FOLDS, COMMON, CONS, LATENT_AE = h1.QC_SCENES, h1.FOLDS, h1.COMMON, h1.CONS, h1.LATENT_AE


def lin(fold=None):
    return {"model": "linear_residual",
            "decoder_path": f"r2://metadata/linear_baseline{'' if fold is None else f'_fold{fold}'}.pt"}


def arms(fold=None):
    return {
        "LR": {**lin(fold)},
        "LRC": {**lin(fold), **CONS},
        "LR1": {**lin(fold), "kernel_mode": "spectral"},
        "M0s": {"model": "hourglass", "kernel_mode": "spectral"},
        "ACs": {**LATENT_AE, **CONS, "kernel_mode": "spectral"},
    }


def cv():
    plan = []
    for arm in arms():
        for i, fold in enumerate(FOLDS):
            plan.append({**COMMON, **arms(i)[arm], "seed": 42, "epochs": 30, "early_stopping_patience": 6,
                         "train_scenes": [s for s in QC_SCENES if s not in fold], "val_scenes": fold,
                         "experiment_name": f"regcv/{arm}/fold{i}"})
    Path("experiments/phase6_cv.json").write_text(json.dumps(plan, indent=2))
    print(f"{len(plan)} CV jobs -> experiments/phase6_cv.json")


def final(record_paths):
    best = {}
    for rp in record_paths:
        for j in json.loads(Path(rp).read_text())["jobs"]:
            if j.get("status") == "done" and j.get("best"):
                best.setdefault(j["experiment_name"].split("/")[1], []).append(j["best"]["epoch"])
    summary, plan = {}, []
    for arm, cfg in arms().items():
        eps = best.get(arm, [])
        if len(eps) != len(FOLDS):
            raise SystemExit(f"arm {arm}: {len(eps)} finished folds, expected {len(FOLDS)}")
        E = int(round(statistics.median(eps)))
        summary[arm] = {"best_epochs": eps, "E": E}
        for seed in (1, 2, 3):
            plan.append({**COMMON, **cfg, "seed": seed, "epochs": E, "early_stopping_patience": 0,
                         "train_scenes": QC_SCENES, "val_scenes": [2],
                         "experiment_name": f"regfinal/{arm}/seed{seed}"})
    Path("experiments/phase6_final.json").write_text(json.dumps(plan, indent=2))
    Path("experiments/phase6_epochs.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"{len(plan)} final jobs -> experiments/phase6_final.json")


if __name__ == "__main__":
    cv() if sys.argv[1] == "cv" else final(sys.argv[2:])
