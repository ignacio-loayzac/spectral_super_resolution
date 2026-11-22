#!/usr/bin/env python
"""Batch-level diagnostic of the raw-Gram perceptual loss (M1) during the first epoch(s).

Rebuilds the exact Study 2 M1 configuration (experiments/phase5_m1_long_seed1.json, seed
overridden) with the local Trainer and records, for every training batch:
  - content term (masked MSE) and weighted raw-Gram term
  - L2 norm of the gradient of each term w.r.t. the SR model parameters
  - prediction scale: RMS of the prediction and of the target over valid pixels (reflectance)
The optimisation step is the normal one (Adam on the summed gradient).

Usage: python scripts/diag_m1_first_epoch.py --seeds 1 2 3 42 --epochs 1 --out results/diag_m1_first_epoch.json
GPU kernels are not bit-identical across devices, so trajectories are not expected to
reproduce the RunPod runs exactly; the point is the mechanism.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from spectral_sr.config import DataConfig, ModelConfig, TrainConfig
from spectral_sr.losses import masked_mse
from spectral_sr.storage import resolve_directory, resolve_path
from spectral_sr.train import Trainer


def build(req, seed, device):
    d = lambda p: resolve_directory(p.replace("r2://", ""))
    mc = ModelConfig(model_type=req["model"], in_channels=7, out_channels=175,
                     encoder_channels=req["encoder_channels"], kernel_mode=req.get("kernel_mode", "spatial"))
    dc = DataConfig(hyperion_train_dir=d(req["hyperion_train_dir"]), hyperion_val_dir=d(req["hyperion_val_dir"]),
                    landsat_train_dir=d(req["landsat_train_dir"]), landsat_val_dir=d(req["landsat_val_dir"]),
                    batch_size=req["batch_size"], num_workers=0, data_scale=req["data_scale"],
                    augment=req["augment"])
    tc = TrainConfig(mode="sr", loss_type=req["loss"], ae_checkpoint=resolve_path(req["ae_checkpoint"]),
                     epochs=1, lr=req["lr"], content_loss_coeff=req["content_loss_coeff"],
                     style_loss_coeff=req["style_loss_coeff"], output_dir="/tmp/diag_m1", seed=seed,
                     gram_mode=req["gram_mode"], ae_kernel_mode=req["ae_kernel_mode"],
                     ae_input_scale=req["ae_input_scale"], use_mask=req["use_mask"], device=device)
    return Trainer(mc, dc, tc)


def gnorm(grads):
    return math.sqrt(sum(float((g ** 2).sum()) for g in grads if g is not None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="experiments/phase5_m1_long_seed1.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 42])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/diag_m1_first_epoch.json")
    args = ap.parse_args()
    req = json.loads(Path(args.plan).read_text())[0]
    out = {}
    for seed in args.seeds:
        t = build(req, seed, args.device)
        t.model.train(); t.loss_fn.train()
        params = [p for p in t.model.parameters() if p.requires_grad]
        rows = []
        for ep in range(args.epochs):
            for b, (x, y) in enumerate(t.train_loader):
                x, y = x.to(t.device), y.to(t.device)
                mask = t._mask(x, y)
                t.optimizer.zero_grad()
                pred = t._predict(x)
                total = t.loss_fn(pred, y, x=x, mask=mask)
                content = t.train_config.content_loss_coeff * masked_mse(pred, y, mask)
                style = total - content
                gc = torch.autograd.grad(content, params, retain_graph=True)
                gs = torch.autograd.grad(style, params, retain_graph=True)
                total.backward()
                t.optimizer.step()
                m = mask.unsqueeze(1).to(pred.dtype)
                n = float(m.sum()) * pred.shape[1]
                scale = t.data_config.data_scale / 1e4
                rows.append({"epoch": ep + 1, "batch": b + 1,
                             "content": float(content), "style": float(style),
                             "grad_content": gnorm(gc), "grad_style": gnorm(gs),
                             "pred_rms": math.sqrt(float((pred.detach() ** 2 * m).sum()) / n) * scale,
                             "target_rms": math.sqrt(float((y ** 2 * m).sum()) / n) * scale})
                if b % 20 == 0:
                    r = rows[-1]
                    print(f"seed {seed} ep {ep+1} batch {b+1:3d} content {r['content']:.3g} style {r['style']:.3g} "
                          f"|g_c| {r['grad_content']:.3g} |g_s| {r['grad_style']:.3g} pred_rms {r['pred_rms']:.3g}", flush=True)
        out[str(seed)] = rows
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
