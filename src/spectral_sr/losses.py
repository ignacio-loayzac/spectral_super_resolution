"""Loss functions for spectral super-resolution with perceptual/style components.

All losses accept an optional boolean ``mask`` of shape (batch, height, width)
that marks valid pixels (see :func:`valid_pixel_mask`); averages are taken over
valid pixels only, so no-data borders do not dominate the loss.

History note (BACKLOG E-M): the 2020 implementation (and v1.0/v1.1 of this
package) captured the autoencoder activations of the *reconstruction* through
forward hooks that called ``.detach()`` and ran under ``torch.no_grad()``.
The Gram term therefore contributed **no gradient** to the generator: models
trained "with perceptual loss" were in practice trained with the content (MSE)
term only. :class:`SpectralSRLoss` fixes this: ground-truth features are
computed without gradient, reconstruction features keep the graph.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def valid_pixel_mask(y: torch.Tensor, x: Optional[torch.Tensor] = None,
                     eps: float = 1e-6) -> torch.Tensor:
    """Boolean (B, H, W) mask of usable pixels.

    A pixel is invalid if its Hyperion spectrum is (numerically) all zeros, or
    if any multispectral band is exactly zero (Landsat no-data fill value).
    """
    valid = y.abs().sum(dim=1) > eps
    if x is not None:
        valid = valid & (x != 0).all(dim=1)
    return valid


def _masked_mean(per_pixel: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean of a (B, H, W) tensor over valid pixels (all pixels if mask is None)."""
    if mask is None:
        return per_pixel.mean()
    m = mask.to(per_pixel.dtype)
    return (per_pixel * m).sum() / m.sum().clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return _masked_mean(((pred - target) ** 2).mean(dim=1), mask)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return _masked_mean((pred - target).abs().mean(dim=1), mask)


def spectral_angle(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel spectral angle in radians, shape (B, H, W)."""
    dot = (pred * target).sum(dim=1)
    norm = pred.norm(dim=1) * target.norm(dim=1)
    cos = (dot / (norm + _EPS)).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos)


def masked_sam(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return _masked_mean(spectral_angle(pred, target), mask)


# ---------------------------------------------------------------------------
# Gram matrices
# ---------------------------------------------------------------------------

def gram_matrix(features: torch.Tensor, mode: str = "raw",
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-sample Gram matrix of a (B, C, H, W) feature map -> (B, C, C).

    mode="raw":          G = F Fᵀ / (C · N)   (as in the thesis; N = valid pixels)
    mode="correlation":  channels are centred and scaled to unit variance over
                         valid pixels, so G is the inter-channel correlation
                         matrix (scale invariant, entries in [-1, 1]).
    """
    b, c, h, w = features.shape
    f = features.reshape(b, c, h * w)
    if mask is not None:
        m = mask.reshape(b, 1, h * w).to(f.dtype)
    else:
        m = torch.ones(b, 1, h * w, dtype=f.dtype, device=f.device)
    n = m.sum(dim=2, keepdim=True).clamp_min(1.0)  # (B, 1, 1)

    if mode == "raw":
        f = f * m
        return torch.bmm(f, f.transpose(1, 2)) / (c * n)
    if mode == "correlation":
        mean = (f * m).sum(dim=2, keepdim=True) / n
        fc = (f - mean) * m
        cov = torch.bmm(fc, fc.transpose(1, 2)) / n
        std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(_EPS))
        return cov / (std.unsqueeze(2) * std.unsqueeze(1))
    raise ValueError(f"unknown gram mode: {mode}")


def gram_loss(feat_pred: torch.Tensor, feat_gt: torch.Tensor, mode: str = "raw",
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.mse_loss(gram_matrix(feat_pred, mode, mask), gram_matrix(feat_gt, mode, mask))


def feature_loss(feat_pred: torch.Tensor, feat_gt: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Perceptual content (feature reconstruction) loss, Johnson et al. 2016."""
    return _masked_mean(((feat_pred - feat_gt) ** 2).mean(dim=1), mask)


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class SpectralSRLoss(nn.Module):
    """Composite loss used for every training variant.

        L = α·content + β·Σ_k w_k·Gram_k + γ·Σ_k w_k·Feat_k + λ·Consistency + η·SAM

    - content: masked MSE or L1 between prediction and ground truth
    - Gram/Feat: computed on the encoder blocks (UB, MB, LB) of a frozen
      autoencoder; gradients flow through the AE into the prediction
    - Consistency: ‖Ŷ·D − X‖² with a fixed 175×7 degradation matrix D
      (predicted hyperspectral pixels must reproduce the multispectral input)
    - SAM: masked mean spectral angle (radians)

    Terms with a zero coefficient are skipped. Call :meth:`components` after
    ``forward`` to get the last value of each term (for logging).
    """

    def __init__(self,
                 content: str = "mse",
                 content_coeff: float = 1.0,
                 perceptual_model: Optional[nn.Module] = None,
                 style_coeff: float = 0.0,
                 feature_coeff: float = 0.0,
                 block_coeff: Sequence[float] = (1.0, 1.0, 1.0),
                 gram_mode: str = "raw",
                 consistency_coeff: float = 0.0,
                 degradation: Optional[torch.Tensor] = None,
                 consistency_bands: Optional[Sequence[int]] = None,
                 sam_coeff: float = 0.0):
        super().__init__()
        if content not in ("mse", "l1"):
            raise ValueError(f"content must be 'mse' or 'l1', got {content}")
        if (style_coeff > 0 or feature_coeff > 0) and perceptual_model is None:
            raise ValueError("style/feature terms need a perceptual_model")
        if consistency_coeff > 0 and degradation is None:
            raise ValueError("consistency term needs a degradation matrix")
        self.content = content
        self.content_coeff = content_coeff
        self.perceptual_model = perceptual_model
        if perceptual_model is not None:
            perceptual_model.eval()
            for p in perceptual_model.parameters():
                p.requires_grad_(False)
        self.style_coeff = style_coeff
        self.feature_coeff = feature_coeff
        self.block_coeff = list(block_coeff)
        self.gram_mode = gram_mode
        self.consistency_coeff = consistency_coeff
        if degradation is not None:
            # (C_hs, C_ms), or (C_hs + 1, C_ms) where the last row is a per-band intercept
            self.register_buffer("degradation", degradation.float())
        else:
            self.degradation = None
        self.consistency_bands = list(consistency_bands) if consistency_bands is not None else None
        self.sam_coeff = sam_coeff
        self._last: Dict[str, float] = {}

    def train(self, mode: bool = True):  # keep the frozen AE in eval mode
        super().train(mode)
        if self.perceptual_model is not None:
            self.perceptual_model.eval()
        return self

    def components(self) -> Dict[str, float]:
        return dict(self._last)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                x: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        terms: Dict[str, torch.Tensor] = {}
        content_fn = masked_mse if self.content == "mse" else masked_l1
        terms["content"] = self.content_coeff * content_fn(pred, target, mask)

        if self.perceptual_model is not None and (self.style_coeff > 0 or self.feature_coeff > 0):
            with torch.no_grad():
                feats_gt = self.perceptual_model.encode_features(target)
            feats_pred = self.perceptual_model.encode_features(pred)  # keeps the graph
            style = pred.new_zeros(())
            feat = pred.new_zeros(())
            for w, fp, fg in zip(self.block_coeff, feats_pred, feats_gt):
                if self.style_coeff > 0:
                    style = style + w * gram_loss(fp, fg, self.gram_mode, mask)
                if self.feature_coeff > 0:
                    feat = feat + w * feature_loss(fp, fg, mask)
            if self.style_coeff > 0:
                terms["style"] = self.style_coeff * style
            if self.feature_coeff > 0:
                terms["feature"] = self.feature_coeff * feat

        if self.consistency_coeff > 0:
            if x is None:
                raise ValueError("consistency term needs the multispectral input x")
            D = self.degradation
            n_hs = pred.shape[1]
            x_hat = torch.einsum("bchw,ck->bkhw", pred, D[:n_hs])
            if D.shape[0] == n_hs + 1:
                x_hat = x_hat + D[n_hs].view(1, -1, 1, 1)
            x_ref = x
            if self.consistency_bands is not None:
                x_hat, x_ref = x_hat[:, self.consistency_bands], x[:, self.consistency_bands]
            terms["consistency"] = self.consistency_coeff * masked_mse(x_hat, x_ref, mask)

        if self.sam_coeff > 0:
            terms["sam"] = self.sam_coeff * masked_sam(pred, target, mask)

        total = sum(terms.values())
        self._last = {k: float(v.detach()) for k, v in terms.items()}
        self._last["total"] = float(total.detach())
        return total


# ---------------------------------------------------------------------------
# Backwards-compatible names
# ---------------------------------------------------------------------------

def GramLoss(perceptual_model: nn.Module, content_loss_coeff: float = 1.0,
             style_loss_coeff: float = 1e-3, block_coeff: List[float] = None,
             device: str = "cpu", **_ignored) -> SpectralSRLoss:
    """Original "MSE + Gram" perceptual loss, with gradients now flowing."""
    return SpectralSRLoss(content="mse", content_coeff=content_loss_coeff,
                          perceptual_model=perceptual_model, style_coeff=style_loss_coeff,
                          block_coeff=block_coeff or (1.0, 1.0, 1.0), gram_mode="raw").to(device)
