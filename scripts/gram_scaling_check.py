#!/usr/bin/env python
"""How each Gram-loss formulation scales when the prediction is multiplied by k.

Uses the frozen 2020 AE (encoder blocks UB/MB/LB, inputs in reflectance with the same
x1e4 input scaling as training) and real test tiles. For a prediction Ŷ = k·Y' (Y' = target with
multiplicative noise) it reports, for k in a log grid, the loss value and ||∇_Ŷ L||:
  raw         ||G(Ŷ) - G(Y)||_F^2 / C^2                     (M1, degree 4 in k)
  frob        ||G(Ŷ) - G(Y)||_F / ||G(Y)||_F               (non-squared, relative; degree 2)
  bures       Bures-Wasserstein distance^2 between G(Ŷ) and G(Y)  (degree 2)
  logeuclid   ||log(G(Ŷ)+εI) - log(G(Y)+εI)||_F^2 / C, ε = 1e-3·tr(G(Y))/C  (grows like log^2 k)
  corr        correlation Gram (M2, degree 0)
Output: results/gram_scaling_check.json
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from spectral_sr.losses import gram_matrix
from spectral_sr.models import Hourglass
from spectral_sr.storage import resolve_directory, resolve_path

torch.manual_seed(0)
EPS = 1e-6


def sqrtm_psd(M):
    w, V = torch.linalg.eigh(M)
    return (V * w.clamp_min(0).sqrt().unsqueeze(-2)) @ V.transpose(-1, -2)


def logm_spd(M):
    w, V = torch.linalg.eigh(M)
    return (V * w.clamp_min(EPS).log().unsqueeze(-2)) @ V.transpose(-1, -2)


def losses(Gp, Gt, Cp, Ct, C):
    I = torch.eye(C, dtype=Gp.dtype)
    # regulariser relative to the TARGET's mean eigenvalue (fixed, independent of the prediction)
    eps_t = (1e-3 * torch.diagonal(Gt, dim1=-2, dim2=-1).mean(-1)).view(-1, 1, 1).detach()
    St = sqrtm_psd(Gt)
    bures = (torch.diagonal(Gp, dim1=-2, dim2=-1).sum(-1) + torch.diagonal(Gt, dim1=-2, dim2=-1).sum(-1)
             - 2 * torch.diagonal(sqrtm_psd(St @ Gp @ St), dim1=-2, dim2=-1).sum(-1)).clamp_min(0)
    return {"raw": ((Gp - Gt) ** 2).sum((-2, -1)) / C ** 2,
            "frob": (Gp - Gt).norm(dim=(-2, -1)) / Gt.norm(dim=(-2, -1)),
            "bures": bures,
            "logeuclid": ((logm_spd(Gp + eps_t * I) - logm_spd(Gt + eps_t * I)) ** 2).sum((-2, -1)) / C,
            "corr": ((Cp - Ct) ** 2).sum((-2, -1)) / C ** 2}


def main():
    ae = Hourglass(175, 175, 150, 100, 75)
    ae.load_state_dict(torch.load(resolve_path("r2://Models/AEHG_150_100_75.pt"), map_location="cpu"))
    ae.eval().double()
    for p in ae.parameters():
        p.requires_grad_(False)
    d = Path(resolve_directory("hyperion_test_npy"))
    files = sorted(d.glob("*.npy"))[::400][:4]
    Y = torch.from_numpy(np.stack([np.load(f) for f in files])).permute(0, 3, 1, 2).double() / 1e4
    Yn = Y * (1 + 0.1 * torch.randn_like(Y))
    feats = lambda z: [f / 1e4 for f in ae.encode_features(z * 1e4)]
    with torch.no_grad():
        ft = feats(Y)
    ks = [0.5, 1, 2, 5, 10, 30, 100, 300, 1000]
    res = {n: {"k": ks, "loss": [], "grad": []} for n in ("raw", "frob", "bures", "logeuclid", "corr")}
    for k in ks:
        P = (k * Yn).clone().requires_grad_(True)
        fp = feats(P)
        tot = {n: 0.0 for n in res}
        for a, b in zip(fp, ft):
            C = a.shape[1]
            L = losses(gram_matrix(a, "raw"), gram_matrix(b, "raw"),
                       gram_matrix(a, "correlation"), gram_matrix(b, "correlation"), C)
            for n in res:
                tot[n] = tot[n] + L[n].mean()
        for n in res:
            g, = torch.autograd.grad(tot[n], P, retain_graph=True)
            res[n]["loss"].append(float(tot[n])); res[n]["grad"].append(float(g.norm()))
    Path("results/gram_scaling_check.json").write_text(json.dumps(res, indent=1))
    for n, r in res.items():
        sl = np.polyfit(np.log(ks[3:]), np.log(np.maximum(r["loss"][3:], 1e-300)), 1)[0]
        sg = np.polyfit(np.log(ks[3:]), np.log(np.maximum(r["grad"][3:], 1e-300)), 1)[0]
        print(f"{n:10s} loss@k=1 {r['loss'][1]:.3g} @k=100 {r['loss'][6]:.3g} | log-log slope loss {sl:.2f}, grad {sg:.2f}")


if __name__ == "__main__":
    main()
