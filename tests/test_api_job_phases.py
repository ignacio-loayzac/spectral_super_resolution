"""/train job exposes phase/progress/timing and uses the shared cache (T-018)."""

from fastapi.testclient import TestClient

from spectral_sr import api


class _FakeTrainer:
    def __init__(self, model_config, data_config, train_config):
        self.cfg = train_config
        self.history = {"train_losses": [], "epoch_seconds": []}
        self.epoch_callback = None

    def train(self):
        for e in range(1, self.cfg.epochs + 1):
            self.history["train_losses"].append(1.0 / e)
            self.history["epoch_seconds"].append(0.5)
            self.epoch_callback(e, self.cfg.epochs, self.history)


def test_train_job_phases(monkeypatch, tmp_path):
    phases, resolved = [], []
    real_update = api._update_job

    def spy_update(job_id, **kw):
        if "phase" in kw:
            phases.append(kw["phase"])
        real_update(job_id, **kw)

    def fake_resolve_directory(prefix, cfg=None, progress=None, **kw):
        resolved.append(prefix)
        progress(10, 10, 1_000_000)
        d = tmp_path / "cache" / prefix
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(api, "_update_job", spy_update)
    monkeypatch.setattr(api, "resolve_directory", fake_resolve_directory)
    monkeypatch.setattr(api, "Trainer", _FakeTrainer)
    monkeypatch.setattr(api, "upload_json", lambda *a, **k: None)
    monkeypatch.setattr(api, "upload_directory", lambda *a, **k: ["exp/x/checkpoint_epoch_final.pt"])
    monkeypatch.delenv("SPECTRAL_RECONSTRUCTION_API_KEY", raising=False)

    client = TestClient(api.app)
    body = {"mode": "sr", "loss": "mse", "epochs": 3,
            "hyperion_train_dir": "r2://hyperion_train_npy", "hyperion_val_dir": "r2://hyperion_val_npy",
            "landsat_train_dir": "r2://landsat_train_npy", "landsat_val_dir": "r2://landsat_val_npy",
            "r2": {"access_key_id": "k", "secret_access_key": "s", "endpoint_url": "http://x", "bucket": "b"}}
    job = client.post("/train", json=body).json()
    final = client.get(f"/jobs/{job['job_id']}").json()

    assert final["status"] == "done", final.get("error")
    assert final["phase"] == "done"
    assert final["started_at"] and final["finished_at"]
    assert final["metrics"]["epoch_seconds"] == [0.5, 0.5, 0.5]
    assert resolved == ["hyperion_train_npy", "hyperion_val_npy", "landsat_train_npy", "landsat_val_npy"]
    assert "downloading:hyperion_train" in phases
    assert "training:epoch 3/3" in phases and phases.index("uploading") > phases.index("training:epoch 3/3")
