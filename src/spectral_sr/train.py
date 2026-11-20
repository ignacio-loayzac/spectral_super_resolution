"""Training logic for spectral super-resolution models."""

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import ModelConfig, DataConfig, TrainConfig
from .datasets import SpectralDataset_npy, HyperspectralDataset_npy
from .losses import SpectralSRLoss, masked_mse, masked_sam, valid_pixel_mask
from .metrics import SSIM
from .models import Hourglass, Koundinya2D_CNN
from .device import default_device


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_hourglass(in_ch: int, out_ch: int, encoder_channels, kernel_mode: str = "spatial") -> Hourglass:
    return Hourglass(in_channels=in_ch, out_channels=out_ch,
                     ub_out_channels=encoder_channels[0], mb_out_channels=encoder_channels[1],
                     lb_out_channels=encoder_channels[2], kernel_mode=kernel_mode)


class Trainer:
    """
    Unified trainer for spectral super-resolution models.

    Supports autoencoder training (Hyperion -> Hyperion) and spectral
    reconstruction (Landsat -> Hyperion) with the losses in
    :class:`~spectral_sr.losses.SpectralSRLoss` (MSE, L1, L1+SAM, perceptual
    with Gram/feature terms, spectral consistency), no-data masking, early
    stopping, a plateau LR scheduler and best-checkpoint selection.

    Args:
        model_config: Model architecture configuration
        data_config: Data loading configuration
        train_config: Training hyperparameters and settings
    """

    def __init__(self,
                 model_config: ModelConfig,
                 data_config: DataConfig,
                 train_config: TrainConfig):
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config

        self.device = torch.device(train_config.device or default_device())
        print(f"Using device: {self.device}")

        set_seed(train_config.seed)

        self.model = self._create_model()
        self.model.to(self.device)

        self.loss_fn = self._create_loss_function()
        self.loss_fn.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=train_config.lr)
        self.scheduler = None
        if train_config.lr_scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min",
                                                                  factor=0.5, patience=3)

        self.ssim = SSIM()

        self.train_loader, self.val_loader = self._create_dataloaders()

        self.history = {
            'train_losses': [],
            'val_losses': [],
            'val_ssim': [],
            'val_sam': [],            # radians, masked
            'val_mse': [],            # masked, in data units (after data_scale)
            'val_rmse_reflectance': [],
            'lr': [],
            'epoch_seconds': [],
            'train_loss_components': [],
        }
        self.best = {"epoch": None, "val_loss": math.inf}
        self.stop_reason = None

        # Optional hook called after every epoch: fn(epoch, total_epochs, history)
        self.epoch_callback = None

    # ------------------------------------------------------------------ setup

    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        if self.model_config.model_type == 'hourglass':
            model = build_hourglass(self.model_config.in_channels, self.model_config.out_channels,
                                    self.model_config.encoder_channels, self.model_config.kernel_mode)
        elif self.model_config.model_type == 'koundinya':
            model = Koundinya2D_CNN()
        else:
            raise ValueError(f"Unknown model type: {self.model_config.model_type}")
        print(f"Created {self.model_config.model_type} model "
              f"({sum(p.numel() for p in model.parameters()):,} parameters)")
        return model

    def _create_loss_function(self) -> nn.Module:
        """Create the composite loss from the configuration."""
        tc = self.train_config
        kwargs = dict(content="mse", content_coeff=tc.content_loss_coeff)

        if tc.loss_type == 'mse':
            pass
        elif tc.loss_type == 'l1':
            kwargs["content"] = "l1"
        elif tc.loss_type == 'l1_sam':
            kwargs["content"] = "l1"
            kwargs["sam_coeff"] = tc.sam_loss_coeff if tc.sam_loss_coeff > 0 else 0.1
        elif tc.loss_type == 'perceptual':
            ae_model = build_hourglass(self.model_config.out_channels, self.model_config.out_channels,
                                       self.model_config.encoder_channels, tc.ae_kernel_mode)
            ae_model.load_state_dict(torch.load(tc.ae_checkpoint, map_location="cpu"))
            ae_model.eval()
            if tc.ae_input_scale != 1.0:
                ae_model = _ScaledInput(ae_model, tc.ae_input_scale)
            print(f"Loaded perceptual autoencoder ({tc.ae_kernel_mode}) from {tc.ae_checkpoint}")
            kwargs.update(perceptual_model=ae_model, style_coeff=tc.style_loss_coeff,
                          feature_coeff=tc.feature_loss_coeff, block_coeff=tc.block_coeff,
                          gram_mode=tc.gram_mode)
        else:
            raise ValueError(f"Unknown loss type: {tc.loss_type}")

        if tc.sam_loss_coeff > 0 and tc.loss_type != 'l1_sam':
            kwargs["sam_coeff"] = tc.sam_loss_coeff
        if tc.consistency_coeff > 0:
            if self.train_config.mode != 'sr':
                raise ValueError("consistency loss only applies to mode='sr'")
            D = torch.from_numpy(np.load(tc.degradation_path)).float()
            n_hs, n_ms = self.model_config.out_channels, self.model_config.in_channels
            if D.shape not in ((n_hs, n_ms), (n_hs + 1, n_ms)):
                raise ValueError(f"degradation matrix has shape {tuple(D.shape)}")
            # D maps reflectance; rescale the intercept if data are not in reflectance units
            if D.shape[0] == n_hs + 1 and self.data_config.data_scale != 10000.0:
                D = D.clone(); D[n_hs] *= 10000.0 / self.data_config.data_scale
            kwargs.update(consistency_coeff=tc.consistency_coeff, degradation=D,
                          consistency_bands=tc.consistency_bands)
        return SpectralSRLoss(**kwargs)

    def _create_dataloaders(self):
        """Create training and validation dataloaders."""
        dc = self.data_config
        if self.train_config.mode == 'autoencoder':
            train_dataset = HyperspectralDataset_npy(dc.hyperion_train_dir, scale=dc.data_scale,
                                                     augment=dc.augment)
            val_dataset = HyperspectralDataset_npy(dc.hyperion_val_dir, scale=dc.data_scale)
        elif self.train_config.mode == 'sr':
            if dc.landsat_train_dir is None or dc.landsat_val_dir is None:
                raise ValueError("landsat_train_dir and landsat_val_dir are required for SR mode")
            train_dataset = SpectralDataset_npy(dc.landsat_train_dir, dc.hyperion_train_dir,
                                                scale=dc.data_scale, augment=dc.augment)
            val_dataset = SpectralDataset_npy(dc.landsat_val_dir, dc.hyperion_val_dir,
                                              scale=dc.data_scale)
        else:
            raise ValueError(f"Unknown mode: {self.train_config.mode}")

        pin = self.device.type == "cuda"
        persistent = dc.num_workers > 0
        train_loader = DataLoader(train_dataset, batch_size=dc.batch_size, shuffle=True,
                                  num_workers=dc.num_workers, pin_memory=pin,
                                  persistent_workers=persistent)
        val_loader = DataLoader(val_dataset, batch_size=dc.batch_size, shuffle=False,
                                num_workers=dc.num_workers, pin_memory=pin,
                                persistent_workers=persistent)
        print(f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
        return train_loader, val_loader

    # --------------------------------------------------------------- helpers

    def _mask(self, x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.train_config.use_mask:
            return None
        return valid_pixel_mask(y, x if self.train_config.mode == 'sr' else None)

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = self.model(x)
        h, w = x.shape[-2:]
        return y_pred.reshape(x.shape[0], self.model_config.out_channels, h, w)

    # -------------------------------------------------------------- training

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        self.loss_fn.train()
        losses, comps = [], {}
        for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            mask = self._mask(x_batch, y_batch)

            self.optimizer.zero_grad()
            y_pred = self._predict(x_batch)
            loss = self.loss_fn(y_pred, y_batch, x=x_batch, mask=mask)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            for k, v in self.loss_fn.components().items():
                comps.setdefault(k, []).append(v)
            if batch_idx % 25 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6g}")
        self.history['train_loss_components'].append({k: float(np.mean(v)) for k, v in comps.items()})
        return float(np.mean(losses))

    def validate(self) -> Dict[str, float]:
        """Validate on the validation set (masked metrics)."""
        self.model.eval()
        self.loss_fn.eval()
        losses, ssims = [], []
        se_sum = ang_sum = n_px = 0.0
        with torch.no_grad():
            for x_val, y_val in self.val_loader:
                x_val = x_val.to(self.device, non_blocking=True)
                y_val = y_val.to(self.device, non_blocking=True)
                mask = self._mask(x_val, y_val)
                y_pred = self._predict(x_val)
                losses.append(self.loss_fn(y_pred, y_val, x=x_val, mask=mask).item())
                ssims.append(self.ssim(y_pred, y_val).item())
                m = mask if mask is not None else torch.ones(y_val.shape[0], *y_val.shape[2:],
                                                              dtype=torch.bool, device=y_val.device)
                npx = float(m.sum())
                se_sum += float(masked_mse(y_pred, y_val, m)) * npx
                ang_sum += float(masked_sam(y_pred, y_val, m)) * npx
                n_px += npx
        mse = se_sum / max(n_px, 1.0)
        # RMSE in reflectance units, whatever data_scale is (stored values are reflectance x 1e4)
        rmse_refl = math.sqrt(mse) * self.data_config.data_scale / 10000.0
        return {'loss': float(np.mean(losses)), 'ssim': float(np.mean(ssims)),
                'sam': ang_sum / max(n_px, 1.0), 'mse': mse, 'rmse_reflectance': rmse_refl}

    def train(self):
        """Main training loop with early stopping, LR scheduling and best checkpoint."""
        tc = self.train_config
        print(f"\nStarting training for up to {tc.epochs} epochs")
        print(f"Mode: {tc.mode}, Loss: {tc.loss_type}, mask={tc.use_mask}, "
              f"early_stopping={tc.early_stopping_patience}, scheduler={tc.lr_scheduler}")
        print("=" * 80)
        self.save_meta()
        epochs_without_improvement = 0

        for epoch in range(tc.epochs):
            print(f"\nEpoch {epoch + 1}/{tc.epochs}")
            print("-" * 80)
            epoch_start = time.perf_counter()

            train_loss = self.train_epoch(epoch)
            val = self.validate()
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val['loss'])
            self.history['val_ssim'].append(val['ssim'])
            self.history['val_sam'].append(val['sam'])
            self.history['val_mse'].append(val['mse'])
            self.history['val_rmse_reflectance'].append(val['rmse_reflectance'])
            self.history['lr'].append(lr)
            self.history['epoch_seconds'].append(round(time.perf_counter() - epoch_start, 2))

            print(f"\nEpoch {epoch + 1} Summary: train {train_loss:.6g} | val {val['loss']:.6g} | "
                  f"RMSE refl {val['rmse_reflectance']:.5f} | SAM {val['sam']:.4f} rad | "
                  f"SSIM {val['ssim']:.4f} | lr {lr:.2e} | {self.history['epoch_seconds'][-1]:.1f} s")

            if val['loss'] < self.best['val_loss']:
                self.best = {"epoch": epoch + 1, "val_loss": val['loss'],
                             **{f"val_{k}": v for k, v in val.items() if k != 'loss'}}
                self.save_checkpoint('best')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if tc.checkpoint_every and (epoch + 1) % tc.checkpoint_every == 0:
                self.save_checkpoint(epoch + 1)

            if self.scheduler is not None:
                self.scheduler.step(val['loss'])

            if self.epoch_callback is not None:
                self.epoch_callback(epoch + 1, tc.epochs, self.history)

            if tc.early_stopping_patience and epochs_without_improvement >= tc.early_stopping_patience:
                self.stop_reason = f"early_stopping (no improvement for {epochs_without_improvement} epochs)"
                print(f"\nStopping: {self.stop_reason}")
                break
        else:
            self.stop_reason = "max_epochs"

        print("\n" + "=" * 80)
        print(f"Training completed ({self.stop_reason}); best epoch {self.best['epoch']}")
        self.history['best'] = self.best
        self.history['stop_reason'] = self.stop_reason
        self.save_checkpoint('final')
        self.save_history()

    # ----------------------------------------------------------------- saving

    def save_meta(self):
        """Write model_meta.json so evaluation can rebuild the model and its scaling."""
        from dataclasses import asdict
        out = Path(self.train_config.output_dir)
        meta = {"model": asdict(self.model_config), "data_scale": self.data_config.data_scale,
                "train": {k: v for k, v in asdict(self.train_config).items() if k != "output_dir"},
                "augment": self.data_config.augment, "batch_size": self.data_config.batch_size}
        (out / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        output_dir = Path(self.train_config.output_dir)
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    def save_history(self):
        """Save training history as JSON."""
        output_dir = Path(self.train_config.output_dir)
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=float)
        print(f"  Saved training history: {history_path}")


class _ScaledInput(nn.Module):
    """Wrap a frozen AE trained on differently scaled data (e.g. the 2020 AE, trained on x1e4 values).

    Inputs are multiplied by ``factor`` and the returned activations divided by it,
    so the raw-Gram and feature terms keep the magnitude they would have in
    reflectance units (otherwise they scale with factor**2 and swamp the content loss).
    """

    def __init__(self, ae: nn.Module, factor: float):
        super().__init__()
        self.ae = ae
        self.factor = factor

    def encode_features(self, x: torch.Tensor):
        return tuple(f / self.factor for f in self.ae.encode_features(x * self.factor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae(x * self.factor)
