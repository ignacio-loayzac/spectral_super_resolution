"""Device auto-detection shared by training, inference and the API."""

from __future__ import annotations

import torch


def default_device() -> str:
    """Return the best available device name: ``cuda``, then ``mps`` (Apple Silicon), then ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
