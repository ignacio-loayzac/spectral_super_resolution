#!/usr/bin/env python
"""
Create the scene-level validation split (BACKLOG T-016) inside R2.

The original dataset has train scenes 1-10 and test scenes 11-15 (tiles are
named ``<scene>_<tile>.npy``). This script holds out ONE whole training scene,
chosen with a fixed seed, so that no scene contributes tiles to both train and
validation. Objects are copied server-side inside the data bucket (nothing is
downloaded), and the original prefixes are left untouched:

    splits/v1/hyperion_train_npy/   splits/v1/landsat_train_npy/
    splits/v1/hyperion_val_npy/     splits/v1/landsat_val_npy/
    splits/v1/manifest.csv          (file, scene, split, source_prefix)
    splits/v1/split.json            (seed, scenes per split, counts)

Test stays at the original ``hyperion_test_npy/`` and ``landsat_test_npy/``.
Re-running is idempotent (objects already copied with the same size are skipped).

Usage:
    python scripts/make_val_split.py --dry-run
    python scripts/make_val_split.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from spectral_sr.storage import R2Config, _client_for_threads, list_prefix

VERSION = "v1"
SEED = 42
SENSORS = ("hyperion", "landsat")


def scene_of(filename: str) -> int:
    return int(filename.split("_", 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only print the plan")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    cfg = R2Config.from_env()
    client = _client_for_threads(cfg, args.workers)
    root = f"splits/{VERSION}"

    listings = {}
    for sensor in SENSORS:
        for split in ("train", "test"):
            prefix = f"{sensor}_{split}_npy/"
            listings[(sensor, split)] = {k[len(prefix):]: s for k, s in list_prefix(prefix, cfg, client=client)}

    train_files = sorted(listings[("hyperion", "train")])
    if sorted(listings[("landsat", "train")]) != train_files:
        print("ERROR: hyperion/landsat train file names differ", file=sys.stderr)
        return 1
    counts = Counter(scene_of(f) for f in train_files)
    val_scene = random.Random(SEED).choice(sorted(counts))
    val_files = [f for f in train_files if scene_of(f) == val_scene]
    tr_files = [f for f in train_files if scene_of(f) != val_scene]
    test_files = sorted(listings[("hyperion", "test")])

    summary = {
        "version": VERSION, "seed": SEED, "rule": "one whole training scene held out for validation",
        "train_scenes": sorted(s for s in counts if s != val_scene), "val_scenes": [val_scene],
        "test_scenes": sorted({scene_of(f) for f in test_files}),
        "n_train": len(tr_files), "n_val": len(val_files), "n_test": len(test_files),
        "val_fraction_of_original_train": round(len(val_files) / len(train_files), 4),
        "prefixes": {f"{s}_{sp}": f"{root}/{s}_{sp}_npy/" for s in SENSORS for sp in ("train", "val")}
                    | {f"{s}_test": f"{s}_test_npy/" for s in SENSORS},
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0

    jobs = []
    for sensor in SENSORS:
        src_prefix = f"{sensor}_train_npy/"
        sizes = listings[(sensor, "train")]
        for split, files in (("train", tr_files), ("val", val_files)):
            dst_prefix = f"{root}/{sensor}_{split}_npy/"
            existing = {k[len(dst_prefix):]: s for k, s in list_prefix(dst_prefix, cfg, client=client)}
            jobs += [(src_prefix + f, dst_prefix + f) for f in files if existing.get(f) != sizes[f]]

    print(f"copying {len(jobs)} objects server-side ...")

    def _copy(job):
        src, dst = job
        client.copy_object(Bucket=cfg.bucket, Key=dst, CopySource={"Bucket": cfg.bucket, "Key": src})

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_copy, j) for j in jobs]):
            fut.result()
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)}")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["file", "scene", "split", "hyperion_prefix", "landsat_prefix"])
    for split, files, pre in (("train", tr_files, f"{root}/{{}}_train_npy/"), ("val", val_files, f"{root}/{{}}_val_npy/"),
                              ("test", test_files, "{}_test_npy/")):
        for f in files:
            w.writerow([f, scene_of(f), split, pre.format("hyperion"), pre.format("landsat")])
    client.put_object(Bucket=cfg.bucket, Key=f"{root}/manifest.csv", Body=buf.getvalue().encode())
    client.put_object(Bucket=cfg.bucket, Key=f"{root}/split.json", Body=json.dumps(summary, indent=2).encode())

    for sensor in SENSORS:  # verify
        for split, files in (("train", tr_files), ("val", val_files)):
            got = len(list_prefix(f"{root}/{sensor}_{split}_npy/", cfg, client=client))
            assert got == len(files), (sensor, split, got, len(files))
    print("OK: split written and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
