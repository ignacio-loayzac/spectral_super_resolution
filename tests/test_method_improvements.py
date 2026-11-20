"""BACKLOG E-M: perceptual gradients, masking, correlation Gram, consistency, augmentation, early stopping."""

import numpy as np
import pytest
import torch

from spectral_sr.config import DataConfig, ModelConfig, TrainConfig
from spectral_sr.datasets import SpectralDataset_npy, _augment_pair
from spectral_sr.losses import SpectralSRLoss, gram_matrix, masked_mse, valid_pixel_mask
from spectral_sr.models import Hourglass
from spectral_sr.train import Trainer


def _ae():
    torch.manual_seed(0)
    return Hourglass(175, 175, 150, 100, 75).eval()


@pytest.mark.parametrize("mode", ["raw", "correlation"])
def test_perceptual_terms_backpropagate(mode):
    """Regression test for the 2020 bug: the Gram term must produce a non-zero gradient."""
    gt = torch.rand(2, 175, 8, 8)
    pred = torch.rand(2, 175, 8, 8, requires_grad=True)
    loss = SpectralSRLoss(content_coeff=1e-12, perceptual_model=_ae(), style_coeff=1.0,
                          feature_coeff=1.0, gram_mode=mode)
    loss(pred, gt).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    ae_params = loss.perceptual_model.parameters()
    assert all(p.grad is None for p in ae_params)  # AE stays frozen


def test_correlation_gram_is_scale_invariant():
    f = torch.randn(2, 6, 5, 5)
    g1, g2 = gram_matrix(f, "correlation"), gram_matrix(3.0 * f + 1.0, "correlation")
    assert torch.allclose(g1, g2, atol=1e-5)
    assert torch.allclose(torch.diagonal(g1, dim1=1, dim2=2), torch.ones(2, 6), atol=1e-5)


def test_mask_excludes_nodata_pixels():
    y = torch.rand(1, 175, 4, 4); x = torch.rand(1, 7, 4, 4)
    y[0, :, 0, 0] = 0.0          # no Hyperion data
    x[0, 3, 1, 1] = 0.0          # Landsat fill value in one band
    m = valid_pixel_mask(y, x)
    assert not m[0, 0, 0] and not m[0, 1, 1] and int(m.sum()) == 14
    pred = y.clone(); pred[0, :, 0, 0] = 100.0  # huge error only on an invalid pixel
    assert float(masked_mse(pred, y, m)) == 0.0
    assert float(masked_mse(pred, y)) > 0.0


def test_consistency_with_intercept_and_band_subset():
    D = torch.zeros(176, 7); D[:175] = torch.rand(175, 7) / 175; D[175] = 0.05
    pred = torch.rand(2, 175, 4, 4)
    x = torch.einsum("bchw,ck->bkhw", pred, D[:175]) + 0.05
    x[:, 0] += 1.0                                  # band 0 inconsistent on purpose
    full = SpectralSRLoss(content_coeff=1e-12, consistency_coeff=1.0, degradation=D)
    subset = SpectralSRLoss(content_coeff=1e-12, consistency_coeff=1.0, degradation=D,
                            consistency_bands=[1, 2, 3, 4, 5, 6])
    assert float(full(pred, pred, x=x)) > 0.1
    assert float(subset(pred, pred, x=x)) < 1e-8


def test_augmentation_keeps_pairs_aligned():
    torch.manual_seed(3)
    a = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    for _ in range(20):
        ax, ay = _augment_pair(a, a.clone())
        assert torch.equal(ax, ay)


def _tiles(root, n, ch, seed=0):
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        np.save(root / f"1_{i}.npy", (rng.random((16, 16, ch)) * 1e4).astype(np.float32))


def test_scaled_dataset(tmp_path):
    _tiles(tmp_path / "ls", 2, 7); _tiles(tmp_path / "hy", 2, 175)
    raw = SpectralDataset_npy(tmp_path / "ls", tmp_path / "hy")[0]
    sc = SpectralDataset_npy(tmp_path / "ls", tmp_path / "hy", scale=1e4)[0]
    assert torch.allclose(raw[1] / 1e4, sc[1]) and sc[1].max() <= 1.0


def test_early_stopping_and_best_checkpoint(tmp_path):
    for s in ("tr", "va"):
        _tiles(tmp_path / f"ls_{s}", 4, 7); _tiles(tmp_path / f"hy_{s}", 4, 175)
    trainer = Trainer(
        ModelConfig(model_type="hourglass", encoder_channels=[8, 8, 8], kernel_mode="spectral"),
        DataConfig(hyperion_train_dir=str(tmp_path / "hy_tr"), hyperion_val_dir=str(tmp_path / "hy_va"),
                   landsat_train_dir=str(tmp_path / "ls_tr"), landsat_val_dir=str(tmp_path / "ls_va"),
                   batch_size=2, num_workers=0, data_scale=1e4, augment=True),
        TrainConfig(mode="sr", loss_type="l1_sam", epochs=30, lr=1e-12, output_dir=str(tmp_path / "out"),
                    checkpoint_every=0, device="cpu", early_stopping_patience=2, lr_scheduler="plateau"),
    )
    trainer.train()
    assert trainer.stop_reason.startswith("early_stopping")
    assert len(trainer.history["val_losses"]) < 30
    assert (tmp_path / "out" / "checkpoint_epoch_best.pt").exists()
    assert (tmp_path / "out" / "model_meta.json").exists()
