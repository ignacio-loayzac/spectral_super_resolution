"""Trainer records per-epoch wall time and calls epoch_callback (T-018 visibility)."""

import numpy as np

from spectral_sr.config import DataConfig, ModelConfig, TrainConfig
from spectral_sr.train import Trainer


def _write_tiles(root, n, channels):
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        np.save(root / f"1_{i}.npy", rng.random((64, 64, channels), dtype=np.float32))


def test_epoch_seconds_and_callback(tmp_path):
    for split in ("train", "val"):
        _write_tiles(tmp_path / f"ls_{split}", 4, 7)
        _write_tiles(tmp_path / f"hy_{split}", 4, 175)
    trainer = Trainer(
        ModelConfig(model_type="hourglass", in_channels=7, out_channels=175, encoder_channels=[150, 100, 75]),
        DataConfig(hyperion_train_dir=str(tmp_path / "hy_train"), hyperion_val_dir=str(tmp_path / "hy_val"),
                   landsat_train_dir=str(tmp_path / "ls_train"), landsat_val_dir=str(tmp_path / "ls_val"),
                   batch_size=2, num_workers=0),
        TrainConfig(mode="sr", loss_type="mse", epochs=2, output_dir=str(tmp_path / "out"),
                    checkpoint_every=10, device="cpu", seed=0),
    )
    calls = []
    trainer.epoch_callback = lambda e, t, h: calls.append((e, t, list(h["epoch_seconds"])))
    trainer.train()
    assert [c[:2] for c in calls] == [(1, 2), (2, 2)]
    assert len(trainer.history["epoch_seconds"]) == 2
    assert all(s > 0 for s in trainer.history["epoch_seconds"])
    assert (tmp_path / "out" / "training_history.json").exists()
