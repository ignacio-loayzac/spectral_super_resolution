"""Cloudflare R2 / S3-compatible storage utilities.

Credentials are resolved in this priority order:
  1. Explicitly passed R2Config object (from API payload or CLI flags)
  2. Environment variables (SPECTRAL_RECONSTRUCTION_R2_ACCESS_KEY_ID, …)
  3. .env file in the current working directory (loaded automatically at import)

No credentials are ever hard-coded here.

r2:// path convention
---------------------
Any path argument that begins with ``r2://`` is treated as an R2 key.
Use ``resolve_path()`` to transparently download it to a local temp file
before passing it to PyTorch / numpy loaders.

Example:
    local = resolve_path("r2://Models/AEHG_150_100_75.pt", cfg)
    model.load_state_dict(torch.load(local))
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import env

logger = logging.getLogger(__name__)

# Load .env on import so env vars are available before anything else reads them
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # don't overwrite already-set env vars
except ImportError:
    pass  # dotenv is optional; real env vars still work fine


# ---------------------------------------------------------------------------
# R2Config
# ---------------------------------------------------------------------------

@dataclass
class R2Config:
    """
    Credentials and targeting configuration for Cloudflare R2.

    Fields default to the corresponding environment variables so callers
    only need to pass what differs from the environment.
    """
    access_key_id: str = field(default_factory=lambda: env.getenv(env.R2_ACCESS_KEY_ID))
    secret_access_key: str = field(default_factory=lambda: env.getenv(env.R2_SECRET_ACCESS_KEY))
    endpoint_url: str = field(default_factory=lambda: env.getenv(env.R2_ENDPOINT_URL))
    bucket: str = field(
        default_factory=lambda: env.getenv(env.R2_BUCKET_NAME, "spectral-reconstruction-data-ena")
    )
    experiments_bucket: str = field(
        default_factory=lambda: env.getenv(
            env.R2_BUCKET_NAME_EXPERIMENTS, "spectral-reconstruction-experiments"
        )
    )

    def for_experiments(self) -> "R2Config":
        """Return a copy targeting the experiments results bucket."""
        return replace(self, bucket=self.experiments_bucket)

    def validate(self):
        """Raise EnvironmentError if any required field is missing."""
        missing = [f for f, v in [
            (f"{env.R2_ACCESS_KEY_ID} / access_key_id", self.access_key_id),
            (f"{env.R2_SECRET_ACCESS_KEY} / secret_access_key", self.secret_access_key),
            (f"{env.R2_ENDPOINT_URL} / endpoint_url", self.endpoint_url),
        ] if not v]
        if missing:
            raise EnvironmentError(
                "Missing R2 credentials: " + ", ".join(missing) + "\n"
                "Set them in your .env file, as environment variables, "
                "or pass them in the request/CLI."
            )

    @classmethod
    def from_env(cls) -> "R2Config":
        """Convenience constructor: read everything from environment / .env."""
        return cls()

    def client(self):
        """Return a configured boto3 S3 client."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required. Install with: uv pip install boto3")

        self.validate()
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",
        )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def is_r2_path(path: str) -> bool:
    """Return True if ``path`` looks like ``r2://some/key``."""
    return isinstance(path, str) and path.startswith("r2://")


def r2_key_from_path(path: str) -> str:
    """Strip the ``r2://`` prefix to get the raw key."""
    return path[len("r2://"):]


ProgressFn = Callable[[int, int, int], None]
"""Progress callback signature: ``fn(files_done, files_total, bytes_done)``."""

_DEFAULT_DOWNLOAD_WORKERS = 64
_COMPLETE_MARKER = ".r2_complete.json"
_MAX_ATTEMPTS = 6


def download_workers() -> int:
    """Parallel download threads (env ``SPECTRAL_RECONSTRUCTION_DOWNLOAD_WORKERS``, default 64)."""
    try:
        return max(1, int(env.getenv(env.DOWNLOAD_WORKERS, str(_DEFAULT_DOWNLOAD_WORKERS))))
    except ValueError:
        return _DEFAULT_DOWNLOAD_WORKERS


def cache_root() -> Path:
    """Root of the local r2:// cache (env ``SPECTRAL_RECONSTRUCTION_CACHE_DIR``)."""
    root = env.getenv(env.CACHE_DIR) or os.path.join(tempfile.gettempdir(), "spectral_sr_cache")
    return Path(root)


def _client_for_threads(cfg: R2Config, max_workers: int):
    """boto3 client whose connection pool can serve ``max_workers`` threads at once."""
    import boto3
    from botocore.config import Config

    cfg.validate()
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name="auto",
        config=Config(max_pool_connections=max_workers + 4,
                      retries={"max_attempts": 8, "mode": "adaptive"}),
    )


@contextmanager
def _file_lock(lock_path: Path):
    """Inter-process/thread exclusive lock (POSIX ``flock``); no-op where unavailable."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as fh:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows
            pass
        try:
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover
                pass


def list_prefix(r2_prefix: str, cfg: Optional[R2Config] = None,
                client=None) -> List[Tuple[str, int]]:
    """Return ``[(key, size), ...]`` for every object under ``r2_prefix``."""
    cfg = cfg or R2Config.from_env()
    client = client or cfg.client()
    out: List[Tuple[str, int]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cfg.bucket, Prefix=r2_prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                out.append((obj["Key"], int(obj["Size"])))
    return out


def download_prefix(r2_prefix: str, local_dir: str, cfg: Optional[R2Config] = None,
                    max_workers: Optional[int] = None,
                    progress: Optional[ProgressFn] = None) -> Tuple[int, int]:
    """
    Download every object under ``r2_prefix`` into ``local_dir`` using a thread pool.

    Files that already exist locally with the same size are skipped, so an
    interrupted download resumes where it left off.

    Returns:
        ``(files_total, bytes_total)``
    """
    cfg = cfg or R2Config.from_env()
    max_workers = max_workers or download_workers()
    client = _client_for_threads(cfg, max_workers)
    prefix = r2_prefix.rstrip("/") + "/"
    objects = list_prefix(prefix, cfg, client=client)
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    total_files = len(objects)
    total_bytes = sum(size for _, size in objects)
    pending = []
    done_files, done_bytes = 0, 0
    for key, size in objects:
        local_file = local_dir_path / key[len(prefix):]
        if local_file.is_file() and local_file.stat().st_size == size:
            done_files += 1
            done_bytes += size
        else:
            pending.append((key, size, local_file))

    logger.info("r2://%s: %d objects (%.1f MB), %d already local, downloading %d with %d threads",
                prefix, total_files, total_bytes / 1e6, done_files, len(pending), max_workers)
    if progress:
        progress(done_files, total_files, done_bytes)

    def _fetch(item):
        # One GET per object (download_file would add a HEAD per object), with
        # explicit retries: R2 occasionally answers 5xx under high concurrency.
        key, size, local_file = item
        local_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_file.with_name(local_file.name + ".part")
        for attempt in range(_MAX_ATTEMPTS):
            try:
                body = client.get_object(Bucket=cfg.bucket, Key=key)["Body"]
                with open(tmp, "wb") as fh:
                    for chunk in iter(lambda: body.read(1 << 20), b""):
                        fh.write(chunk)
                if tmp.stat().st_size != size:
                    raise IOError(f"size mismatch for {key}: {tmp.stat().st_size} != {size}")
                os.replace(tmp, local_file)
                return size
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                delay = min(30.0, 0.5 * 2 ** attempt) * (0.5 + random.random())
                logger.warning("r2://%s attempt %d failed (%s); retrying in %.1fs",
                               key, attempt + 1, exc, delay)
                time.sleep(delay)

    lock = threading.Lock()
    last_report = [0.0]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch, item) for item in pending]
        for fut in as_completed(futures):
            size = fut.result()  # re-raises download errors
            with lock:
                done_files += 1
                done_bytes += size
                now = time.monotonic()
                if progress and (now - last_report[0] > 2.0 or done_files == total_files):
                    last_report[0] = now
                    progress(done_files, total_files, done_bytes)

    logger.info("r2://%s: %d files ready in %s", prefix, total_files, local_dir)
    return total_files, total_bytes


def resolve_path(path: str, cfg: Optional[R2Config] = None,
                 tmp_dir: Optional[str] = None, use_cache: bool = True) -> str:
    """
    Resolve a path that may be local or an ``r2://`` reference.

    Local paths are returned unchanged. ``r2://key`` objects are downloaded
    once into the local cache (see :func:`cache_root`) and reused afterwards;
    with ``use_cache=False`` they go to a fresh temp file in ``tmp_dir``.
    """
    if not is_r2_path(path):
        return path

    key = r2_key_from_path(path)
    cfg = cfg or R2Config.from_env()
    client = cfg.client()

    if use_cache:
        local = cache_root() / cfg.bucket / key
        with _file_lock(local.with_name(local.name + ".lock")):
            size = int(client.head_object(Bucket=cfg.bucket, Key=key)["ContentLength"])
            if local.is_file() and local.stat().st_size == size:
                logger.info("Cache hit r2://%s -> %s", key, local)
                return str(local)
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_name(local.name + ".part")
            logger.info("Downloading r2://%s -> %s", key, local)
            client.download_file(cfg.bucket, key, str(tmp))
            os.replace(tmp, local)
            return str(local)

    suffix = Path(key).suffix or ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir)
    tmp.close()
    logger.info("Downloading r2://%s -> %s", key, tmp.name)
    client.download_file(cfg.bucket, key, tmp.name)
    return tmp.name


def resolve_directory(r2_prefix: str, cfg: Optional[R2Config] = None,
                      local_dir: Optional[str] = None,
                      progress: Optional[ProgressFn] = None,
                      max_workers: Optional[int] = None) -> str:
    """
    Make every object under an R2 prefix available in a local directory.

    With ``local_dir=None`` (recommended) the files go to the shared local
    cache ``<cache_root>/<bucket>/<prefix>`` and are reused by later calls:
    a completed download leaves a marker file, and the next call only checks
    that the object count still matches before returning immediately.
    Concurrent callers for the same prefix are serialised with a file lock.

    Passing ``local_dir`` forces a (parallel, resumable) download there.

    Returns:
        Local directory path string
    """
    cfg = cfg or R2Config.from_env()
    prefix = r2_prefix.rstrip("/")

    if local_dir is not None:
        download_prefix(prefix, local_dir, cfg, max_workers=max_workers, progress=progress)
        return local_dir

    target = cache_root() / cfg.bucket / prefix
    marker = target / _COMPLETE_MARKER
    with _file_lock(target.parent / f".{target.name}.lock"):
        if marker.is_file():
            try:
                meta = json.loads(marker.read_text())
                n_local = sum(1 for p in target.rglob("*")
                              if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".part"))
                if n_local == meta.get("files"):
                    logger.info("Cache hit r2://%s -> %s (%d files)", prefix, target, n_local)
                    if progress:
                        progress(n_local, n_local, int(meta.get("bytes", 0)))
                    return str(target)
            except (ValueError, OSError):
                pass
            marker.unlink(missing_ok=True)
        n_files, n_bytes = download_prefix(prefix, str(target), cfg,
                                           max_workers=max_workers, progress=progress)
        marker.write_text(json.dumps({"prefix": prefix, "bucket": cfg.bucket,
                                      "files": n_files, "bytes": n_bytes,
                                      "completed_at": time.time()}))
    return str(target)


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def upload_file(local_path: Path, r2_key: str,
                cfg: Optional[R2Config] = None) -> str:
    """Upload a local file to R2 and return its key."""
    cfg = cfg or R2Config.from_env()
    local_path = Path(local_path)
    logger.info("Uploading %s -> r2://%s", local_path, r2_key)
    cfg.client().upload_file(str(local_path), cfg.bucket, r2_key)
    logger.info("Upload complete: %s", r2_key)
    return r2_key


def upload_bytes(data: bytes, r2_key: str,
                 content_type: str = "application/octet-stream",
                 cfg: Optional[R2Config] = None) -> str:
    """Upload raw bytes to R2."""
    cfg = cfg or R2Config.from_env()
    logger.info("Uploading bytes -> r2://%s", r2_key)
    cfg.client().put_object(
        Bucket=cfg.bucket,
        Key=r2_key,
        Body=io.BytesIO(data),
        ContentType=content_type,
    )
    logger.info("Upload complete: %s", r2_key)
    return r2_key


def upload_json(obj: dict, r2_key: str,
                cfg: Optional[R2Config] = None) -> str:
    """Upload a Python dict as a pretty-printed JSON file to R2."""
    data = json.dumps(obj, indent=2).encode("utf-8")
    return upload_bytes(data, r2_key, content_type="application/json", cfg=cfg)


def upload_directory(local_dir: Path, r2_prefix: str,
                     cfg: Optional[R2Config] = None,
                     extensions: tuple = (".pt", ".json", ".png", ".npy", ".csv")) -> List[str]:
    """
    Recursively upload all files under ``local_dir`` to R2.

    Args:
        local_dir:  Local source directory
        r2_prefix:  Key prefix (folder) inside the bucket
        cfg:        R2Config (falls back to env vars)
        extensions: Whitelist of file extensions (``None`` = all)

    Returns:
        List of uploaded R2 keys
    """
    cfg = cfg or R2Config.from_env()
    local_dir = Path(local_dir)
    uploaded: List[str] = []

    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if extensions and file_path.suffix.lower() not in extensions:
            continue

        relative = file_path.relative_to(local_dir)
        r2_key = f"{r2_prefix}/{relative}".replace("\\", "/")
        upload_file(file_path, r2_key, cfg=cfg)
        uploaded.append(r2_key)

    return uploaded


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_experiments(cfg: Optional[R2Config] = None) -> List[str]:
    """Return top-level experiment folder names in the experiments bucket."""
    cfg = cfg or R2Config.from_env()
    response = cfg.client().list_objects_v2(
        Bucket=cfg.experiments_bucket, Delimiter="/"
    )
    return [cp["Prefix"].rstrip("/") for cp in response.get("CommonPrefixes", [])]


def list_keys(prefix: str = "", cfg: Optional[R2Config] = None) -> List[str]:
    """List all object keys under an optional prefix."""
    cfg = cfg or R2Config.from_env()
    paginator = cfg.client().get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=cfg.bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys
