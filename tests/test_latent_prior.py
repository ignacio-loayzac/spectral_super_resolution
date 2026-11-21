"""H1/H5: latent-space SR with a frozen spectral decoder; scene filters."""

import numpy as np
import torch

from spectral_sr.config import DataConfig, ModelConfig, TrainConfig
from spectral_sr.datasets import SpectralDataset_npy
from spectral_sr.models import BottleneckSpectralAE, LatentSR, PCADecoder
from spectral_sr.train import Trainer


def test_bottleneck_has_no_bypass():
    ae = BottleneckSpectralAE(175, 16)
    z = ae.encode(torch.rand(2, 175, 4, 4))
    assert z.shape == (2, 16, 4, 4)
    assert ae(torch.rand(2, 175, 4, 4)).shape == (2, 175 * 16)


def test_latent_sr_ae_decoder_frozen_and_roundtrip(tmp_path):
    ae = BottleneckSpectralAE(175, 16)
    torch.save(ae.state_dict(), tmp_path / "ae.pt")
    m = LatentSR(7, 175, [8, 8, 8], latent_dim=16, decoder_type="ae", decoder_path=str(tmp_path / "ae.pt"))
    assert all(not p.requires_grad for p in m.decoder.parameters())
    out = m(torch.rand(2, 7, 8, 8))
    assert out.shape == (2, 175 * 64)
    out.sum().backward()
    assert any(p.grad is not None for p in m.generator.parameters())
    # rebuild without decoder_path, load the full checkpoint -> identical outputs
    m2 = LatentSR(7, 175, [8, 8, 8], latent_dim=16, decoder_type="ae")
    m2.load_state_dict(m.state_dict())
    x = torch.rand(1, 7, 8, 8)
    assert torch.allclose(m.eval()(x), m2.eval()(x))


def test_pca_decoder(tmp_path):
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(175, 4)))[0].T
    mean = rng.random(175)
    np.savez(tmp_path / "pca.npz", mean=mean, basis=basis)
    dec = PCADecoder.from_file(str(tmp_path / "pca.npz"))
    z = torch.zeros(1, 4, 2, 2)
    assert torch.allclose(dec(z)[0, :, 0, 0], torch.from_numpy(mean).float(), atol=1e-6)
    m = LatentSR(7, 175, [8, 8, 8], latent_dim=4, decoder_type="pca", decoder_path=str(tmp_path / "pca.npz"))
    m2 = LatentSR(7, 175, [8, 8, 8], latent_dim=4, decoder_type="pca")
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2.decoder.basis, m.decoder.basis)


def _tiles(root, scenes, ch):
    root.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for sc in scenes:
        for i in range(2):
            np.save(root / f"{sc}_{i}.npy", (rng.random((16, 16, ch)) * 1e4).astype(np.float32))


def test_scene_filter_and_latent_training(tmp_path):
    _tiles(tmp_path / "ls", [1, 2, 3], 7); _tiles(tmp_path / "hy", [1, 2, 3], 175)
    ds = SpectralDataset_npy(tmp_path / "ls", tmp_path / "hy", scenes=[1, 3])
    assert sorted({p.name.split("_")[0] for p in ds.hsi_path_list}) == ["1", "3"]
    ae = BottleneckSpectralAE(175, 16)
    torch.save(ae.state_dict(), tmp_path / "ae.pt")
    t = Trainer(ModelConfig(model_type="latent", encoder_channels=[8, 8, 8], latent_dim=16,
                            decoder_type="ae", decoder_path=str(tmp_path / "ae.pt")),
                DataConfig(hyperion_train_dir=str(tmp_path / "hy"), hyperion_val_dir=str(tmp_path / "hy"),
                           landsat_train_dir=str(tmp_path / "ls"), landsat_val_dir=str(tmp_path / "ls"),
                           batch_size=2, num_workers=0, data_scale=1e4, train_scenes=[1, 3], val_scenes=[2]),
                TrainConfig(mode="sr", epochs=1, output_dir=str(tmp_path / "out"), checkpoint_every=0, device="cpu"))
    assert len(t.train_loader.dataset) == 4 and len(t.val_loader.dataset) == 2
    dec_before = [p.clone() for p in t.model.decoder.parameters()]
    t.train()
    assert all(torch.equal(a, b) for a, b in zip(dec_before, t.model.decoder.parameters()))
