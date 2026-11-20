"""Configuration classes for training and inference."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal


@dataclass
class ModelConfig:
    """
    Configuration for model architecture.
    
    Args:
        model_type: Type of model ('hourglass' or 'koundinya')
        in_channels: Number of input channels (7 for Landsat, 175 for Hyperion)
        out_channels: Number of output channels (175 for Hyperion)
        encoder_channels: List of encoder output channels [upper, middle, lower]
                         E.g., [150, 100, 75] or [90, 45, 10]
    """
    model_type: Literal['hourglass', 'koundinya'] = 'hourglass'
    in_channels: int = 7
    out_channels: int = 175
    encoder_channels: List[int] = field(default_factory=lambda: [150, 100, 75])
    kernel_mode: Literal['spatial', 'spectral'] = 'spatial'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.model_type == 'koundinya':
            if self.in_channels != 7 or self.out_channels != 175:
                raise ValueError(
                    "Koundinya2D_CNN only supports 7->175 channels (fixed architecture)"
                )
        
        if self.model_type == 'hourglass':
            if len(self.encoder_channels) != 3:
                raise ValueError(
                    f"encoder_channels must have exactly 3 values, got {len(self.encoder_channels)}"
                )


@dataclass
class DataConfig:
    """
    Configuration for data loading.
    
    Args:
        hyperion_train_dir: Path to Hyperion training patches (.npy)
        hyperion_val_dir: Path to Hyperion validation patches (.npy)
        hyperion_test_dir: Optional path to Hyperion test patches (.npy)
        landsat_train_dir: Optional path to Landsat training patches (.npy)
        landsat_val_dir: Optional path to Landsat validation patches (.npy)
        landsat_test_dir: Optional path to Landsat test patches (.npy)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    """
    hyperion_train_dir: str
    hyperion_val_dir: str
    hyperion_test_dir: Optional[str] = None
    landsat_train_dir: Optional[str] = None
    landsat_val_dir: Optional[str] = None
    landsat_test_dir: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    data_scale: float = 1.0      # divide stored values by this (10000 -> reflectance)
    augment: bool = False        # random rot90/flip on the training set
    
    def __post_init__(self):
        """Validate paths exist."""
        for path_attr in ['hyperion_train_dir', 'hyperion_val_dir', 
                          'landsat_train_dir', 'landsat_val_dir']:
            path_val = getattr(self, path_attr)
            if path_val is not None:
                path = Path(path_val)
                if not path.exists():
                    raise ValueError(f"{path_attr} does not exist: {path_val}")


@dataclass
class TrainConfig:
    """
    Configuration for training.
    
    Args:
        mode: Training mode ('autoencoder' or 'sr' for spectral reconstruction)
        loss_type: Loss function ('mse' or 'perceptual')
        ae_checkpoint: Path to pretrained autoencoder (required for perceptual loss)
        epochs: Number of training epochs
        lr: Learning rate
        content_loss_coeff: Weight for content loss (alpha, default 1.0)
        style_loss_coeff: Weight for style/perceptual loss (beta, default 1e-3)
        block_coeff: Weights for encoder blocks [upper, middle, lower]
        output_dir: Directory to save checkpoints and logs
        checkpoint_every: Save checkpoint every N epochs
        device: Device to use ('cuda', 'mps' or 'cpu'; None for auto-detect)
        seed: Random seed for reproducibility
    """
    mode: Literal['autoencoder', 'sr'] = 'sr'
    loss_type: Literal['mse', 'l1', 'l1_sam', 'perceptual'] = 'mse'
    ae_checkpoint: Optional[str] = None
    epochs: int = 5
    lr: float = 1e-3
    content_loss_coeff: float = 1.0
    style_loss_coeff: float = 1e-3
    block_coeff: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    output_dir: str = './outputs'
    checkpoint_every: int = 1
    device: Optional[str] = None
    seed: int = 42
    # --- BACKLOG E-M improvements ---
    gram_mode: Literal['raw', 'correlation'] = 'raw'
    feature_loss_coeff: float = 0.0          # perceptual content term (gamma)
    sam_loss_coeff: float = 0.0              # used by loss_type='l1_sam' (defaults to 0.1 there)
    consistency_coeff: float = 0.0           # spectral consistency term (lambda)
    degradation_path: Optional[str] = None   # .npy (175x7, or 176x7 with intercept row)
    consistency_bands: Optional[List[int]] = None  # OLI bands used by the consistency term (None = all)
    ae_kernel_mode: Literal['spatial', 'spectral'] = 'spatial'
    ae_input_scale: float = 1.0              # multiply AE inputs (e.g. 10000 for the 2020 AE with scaled data)
    use_mask: bool = True                    # ignore no-data pixels in loss and metrics
    early_stopping_patience: int = 0         # 0 disables early stopping
    lr_scheduler: Literal['none', 'plateau'] = 'none'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.loss_type == 'perceptual' and self.ae_checkpoint is None:
            raise ValueError(
                "ae_checkpoint is required when using perceptual loss"
            )
        
        if self.consistency_coeff > 0 and self.degradation_path is None:
            raise ValueError("degradation_path is required when consistency_coeff > 0")
        if self.ae_checkpoint is not None:
            ae_path = Path(self.ae_checkpoint)
            if not ae_path.exists():
                raise ValueError(f"ae_checkpoint does not exist: {self.ae_checkpoint}")
        
        if len(self.block_coeff) != 3:
            raise ValueError(
                f"block_coeff must have exactly 3 values, got {len(self.block_coeff)}"
            )
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """
    Configuration for inference/reconstruction.
    
    Args:
        checkpoint: Path to trained model checkpoint
        input_dir: Directory with input multispectral .npy patches
        output_dir: Directory to save reconstructed patches
        ground_truth_dir: Optional directory with ground truth for metrics
        device: Device to use ('cuda', 'mps' or 'cpu'; None for auto-detect)
    """
    checkpoint: str
    input_dir: str
    output_dir: str
    ground_truth_dir: Optional[str] = None
    device: Optional[str] = None
    
    def __post_init__(self):
        """Validate paths."""
        if not Path(self.checkpoint).exists():
            raise ValueError(f"Checkpoint does not exist: {self.checkpoint}")
        
        if not Path(self.input_dir).exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.ground_truth_dir is not None:
            if not Path(self.ground_truth_dir).exists():
                raise ValueError(
                    f"Ground truth directory does not exist: {self.ground_truth_dir}"
                )
