"""Parallel / cached R2 download (T-018) against an in-memory fake S3 client."""

import threading

import pytest

from spectral_sr import storage
from spectral_sr.storage import R2Config


class FakeS3:
    """Minimal stand-in for the boto3 S3 client calls used by storage.py."""

    def __init__(self, objects):
        self.objects = dict(objects)  # key -> bytes
        self.downloads = 0
        self.fail_once = set()
        self._lock = threading.Lock()

    def get_paginator(self, _name):
        objects = self.objects

        class _Pag:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v)}
                                    for k, v in sorted(objects.items()) if k.startswith(Prefix)]}
        return _Pag()

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}

    def download_file(self, bucket, key, filename):
        with self._lock:
            self.downloads += 1
        with open(filename, "wb") as fh:
            fh.write(self.objects[key])

    def get_object(self, Bucket, Key):
        import io
        with self._lock:
            self.downloads += 1
            if Key in self.fail_once:
                self.fail_once.discard(Key)
                raise RuntimeError("500 Internal Server Error")
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    objs = {f"tiles/{i}_{j}.npy": bytes([i]) * (100 + j) for i in range(3) for j in range(20)}
    objs["tiles_other/x.npy"] = b"must-not-be-downloaded"
    objs["Models/ae.pt"] = b"weights" * 10
    client = FakeS3(objs)
    monkeypatch.setattr(storage, "_client_for_threads", lambda cfg, n: client)
    monkeypatch.setattr(R2Config, "client", lambda self: client)
    monkeypatch.setenv("SPECTRAL_RECONSTRUCTION_CACHE_DIR", str(tmp_path / "cache"))
    cfg = R2Config(access_key_id="k", secret_access_key="s", endpoint_url="http://x", bucket="b")
    return client, cfg, tmp_path


def test_download_prefix_parallel_and_resume(fake):
    client, cfg, tmp = fake
    out = tmp / "out"
    seen = []
    n, nbytes = storage.download_prefix("tiles", str(out), cfg, max_workers=8,
                                        progress=lambda d, t, b: seen.append((d, t)))
    assert n == 60 and client.downloads == 60
    assert sorted(p.name for p in out.iterdir()) == sorted(k.split("/")[1] for k in client.objects if k.startswith("tiles/"))
    assert not (out / "x.npy").exists()  # 'tiles_other/' must not match prefix 'tiles'
    assert seen[-1] == (60, 60)
    # Resume: delete two files, corrupt one -> only those 3 are fetched again
    (out / "0_0.npy").unlink(); (out / "1_1.npy").unlink(); (out / "2_2.npy").write_bytes(b"x")
    storage.download_prefix("tiles", str(out), cfg, max_workers=8)
    assert client.downloads == 63
    assert not list(out.glob("*.part"))


def test_resolve_directory_uses_cache(fake):
    client, cfg, tmp = fake
    d1 = storage.resolve_directory("tiles/", cfg=cfg)
    assert client.downloads == 60
    d2 = storage.resolve_directory("tiles", cfg=cfg)  # trailing slash irrelevant
    assert d1 == d2 and client.downloads == 60     # cache hit, nothing downloaded
    assert d1.startswith(str(tmp / "cache"))
    # A missing file invalidates the marker -> only the missing file is re-downloaded
    next(p for p in __import__("pathlib").Path(d1).glob("*.npy")).unlink()
    storage.resolve_directory("tiles", cfg=cfg)
    assert client.downloads == 61


def test_resolve_path_cached(fake):
    client, cfg, _ = fake
    a = storage.resolve_path("r2://Models/ae.pt", cfg)
    b = storage.resolve_path("r2://Models/ae.pt", cfg)
    assert a == b and client.downloads == 1
    assert open(a, "rb").read() == client.objects["Models/ae.pt"]
    assert storage.resolve_path("/local/file.pt", cfg) == "/local/file.pt"


def test_transient_errors_are_retried(fake, monkeypatch):
    client, cfg, tmp = fake
    monkeypatch.setattr(storage.time, "sleep", lambda s: None)
    client.fail_once = {"tiles/0_3.npy", "tiles/2_7.npy"}
    n, _ = storage.download_prefix("tiles", str(tmp / "retry"), cfg, max_workers=8)
    assert n == 60 and client.downloads == 62
    assert (tmp / "retry" / "0_3.npy").read_bytes() == client.objects["tiles/0_3.npy"]
