#!/usr/bin/env python
"""
Scene-level quality control of the OLI/Hyperion pairs (BACKLOG T-091) and split v2.

For every scene, Hyperion is band-averaged over the nominal OLI band edges (boxcar
SRF) and compared with the co-registered OLI pixels (valid pixels only). A scene
FAILS when the Pearson correlation in Red OR NIR is below --min-r (default 0.5):
for well co-registered, near-simultaneous surface reflectance pairs these two bands
correlate strongly (r > 0.7 in every passing scene), so r < 0.5 indicates
mis-registration, clouds or a temporal mismatch -> unusable as supervision.

Failing TRAIN scenes are excluded in split v2 (server-side copy in R2):
    splits/v2/{hyperion,landsat}_train_npy/   (train v1 minus failing scenes)
Validation (splits/v1 *_val_npy) and test (original prefixes) are unchanged;
their QC is reported but never used to drop data.

Outputs data/qc_scenes.csv, data/qc_summary.json (+ r2://metadata/ and splits/v2/).
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from spectral_sr.storage import R2Config, _client_for_threads, list_prefix, resolve_directory

OLI_EDGES_NM = [(433, 453), (450, 515), (525, 600), (630, 680), (845, 885), (1560, 1660), (2100, 2300)]
OLI_NAMES = ["CA", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
SPLITS = {"train": ("splits/v1/hyperion_train_npy", "splits/v1/landsat_train_npy"),
          "val": ("splits/v1/hyperion_val_npy", "splits/v1/landsat_val_npy"),
          "test": ("hyperion_test_npy", "landsat_test_npy")}


def boxcar(wl):
    B = np.zeros((175, 7))
    for k, (a, b) in enumerate(OLI_EDGES_NM):
        sel = (wl >= a) & (wl <= b)
        B[sel, k] = 1.0 / sel.sum()
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-r", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = R2Config.from_env()
    B = boxcar(np.load("data/wavelengths_175.npy"))

    rows = []
    for split, (h, l) in SPLITS.items():
        hd, ld = resolve_directory(h), resolve_directory(l)
        by = collections.defaultdict(list)
        for p in sorted(Path(hd).glob("*.npy")):
            by[int(p.name.split("_")[0])].append(p.name)
        for sc in sorted(by):
            Y = np.concatenate([np.load(Path(hd) / n).reshape(-1, 175) for n in by[sc]]).astype(np.float64) / 1e4
            X = np.concatenate([np.load(Path(ld) / n).reshape(-1, 7) for n in by[sc]]).astype(np.float64) / 1e4
            v = (np.abs(Y).sum(1) > 1e-6) & (X != 0).all(1)
            Z = Y[v] @ B
            Xv = X[v]
            r = [float(np.corrcoef(Xv[:, k], Z[:, k])[0, 1]) for k in range(7)]
            ratio = [float(Xv[:, k].mean() / max(Z[:, k].mean(), 1e-4)) for k in range(7)]
            passed = r[3] >= args.min_r and r[4] >= args.min_r
            rows.append({"scene": sc, "split": split, "tiles": len(by[sc]), "invalid_pct": round(100 * (1 - v.mean()), 2),
                         **{f"r_{n}": round(x, 3) for n, x in zip(OLI_NAMES, r)},
                         **{f"ratio_{n}": round(x, 2) for n, x in zip(OLI_NAMES, ratio)},
                         "qc_pass": passed})
            print(f"scene {sc:2d} {split:5s} r(Red)={r[3]:.2f} r(NIR)={r[4]:.2f} -> {'PASS' if passed else 'FAIL'}")

    excluded = sorted(r["scene"] for r in rows if r["split"] == "train" and not r["qc_pass"])
    summary = {"criterion": f"Pearson r(OLI, boxcar Hyperion) >= {args.min_r} in Red and NIR",
               "excluded_train_scenes": excluded,
               "failing_val_scenes": sorted(r["scene"] for r in rows if r["split"] == "val" and not r["qc_pass"]),
               "failing_test_scenes": sorted(r["scene"] for r in rows if r["split"] == "test" and not r["qc_pass"]),
               "n_train_v2": sum(r["tiles"] for r in rows if r["split"] == "train" and r["qc_pass"]),
               "prefixes_v2": {"hyperion_train": "splits/v2/hyperion_train_npy/", "landsat_train": "splits/v2/landsat_train_npy/",
                               "hyperion_val": "splits/v1/hyperion_val_npy/", "landsat_val": "splits/v1/landsat_val_npy/",
                               "hyperion_test": "hyperion_test_npy/", "landsat_test": "landsat_test_npy/"}}
    print(json.dumps(summary, indent=2))

    Path("data").mkdir(exist_ok=True)
    with open("data/qc_scenes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    Path("data/qc_summary.json").write_text(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    client = _client_for_threads(cfg, 64)
    jobs = []
    for sensor in ("hyperion", "landsat"):
        src, dst = f"splits/v1/{sensor}_train_npy/", f"splits/v2/{sensor}_train_npy/"
        existing = {k[len(dst):] for k, _ in list_prefix(dst, cfg, client=client)}
        for key, _ in list_prefix(src, cfg, client=client):
            name = key[len(src):]
            if int(name.split("_")[0]) not in excluded and name not in existing:
                jobs.append((key, dst + name))
    print(f"copying {len(jobs)} objects to splits/v2 ...")
    with ThreadPoolExecutor(64) as ex:
        for f in as_completed([ex.submit(client.copy_object, Bucket=cfg.bucket, Key=d,
                                         CopySource={"Bucket": cfg.bucket, "Key": s}) for s, d in jobs]):
            f.result()
    for name in ("qc_scenes.csv", "qc_summary.json"):
        client.upload_file(f"data/{name}", cfg.bucket, f"metadata/{name}")
    client.put_object(Bucket=cfg.bucket, Key="splits/v2/split.json", Body=json.dumps(summary, indent=2).encode())
    for sensor in ("hyperion", "landsat"):
        n = len(list_prefix(f"splits/v2/{sensor}_train_npy/", cfg, client=client))
        assert n == summary["n_train_v2"], (sensor, n)
    print("OK: splits/v2 written and verified")


if __name__ == "__main__":
    main()
