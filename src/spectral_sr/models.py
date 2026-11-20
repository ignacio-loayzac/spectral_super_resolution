"""Neural network architectures for spectral super-resolution."""

import torch
import torch.nn as nn


class Hourglass(nn.Module):
    """
    Hourglass architecture for spectral reconstruction.
    
    Symmetric UNet-like architecture that compresses along the spectral axis only.
    Can be used for:
    - Autoencoder: in_channels == out_channels (e.g., 175 -> 175 for Hyperion)
    - Spectral Reconstruction: in_channels != out_channels (e.g., 7 -> 175 for Landsat to Hyperion)
    
    Args:
        in_channels: Number of input channels/bands
        out_channels: Number of output channels/bands
        ub_out_channels: Output channels for upper block encoder
        mb_out_channels: Output channels for middle block encoder
        lb_out_channels: Output channels for lower block encoder (bottleneck)
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 ub_out_channels: int, mb_out_channels: int, lb_out_channels: int,
                 kernel_mode: str = "spatial"):
        super(Hourglass, self).__init__()
        if kernel_mode not in ("spatial", "spectral"):
            raise ValueError(f"kernel_mode must be 'spatial' or 'spectral', got {kernel_mode}")
        # "spatial": 3x3 / 5x5 kernels (2020 architecture).
        # "spectral": every convolution is 1x1, i.e. a per-pixel spectral network.
        self.kernel_mode = kernel_mode
        k3, p3, k5, p5 = (3, 1, 5, 2) if kernel_mode == "spatial" else (1, 0, 1, 0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoder_upperBlock = self._conv_block(in_channels, ub_out_channels, stride2=1,
                                                    kernel_size=k3, padding=p3)
        self.encoder_middleBlock = self._conv_block(ub_out_channels, mb_out_channels, 
                                                     stride2=1, kernel_size=k5, padding=p5)
        self.encoder_lowerBlock = self._conv_block_single(mb_out_channels, lb_out_channels,
                                                          kernel_size=k3, padding=p3)
        
        # Skip connections
        self.skip2 = nn.Conv2d(ub_out_channels, mb_out_channels, kernel_size=1)
        self.skip1 = nn.Conv2d(mb_out_channels, lb_out_channels, kernel_size=1)
        # Accept checkpoints saved by legacy/modelos.py (2020), where each skip
        # connection was wrapped in nn.Sequential ("skip1.0.weight" vs "skip1.weight").
        self._register_load_state_dict_pre_hook(self._remap_legacy_keys)
        
        # Decoder
        self.decoder_lowerBlock = self._conv_block_single(lb_out_channels, lb_out_channels,
                                                          kernel_size=k3, padding=p3)
        self.decoder_middleBlock = self._conv_block(lb_out_channels, mb_out_channels, 
                                                     stride2=1, upsample=False, 
                                                     kernel_size=k5, padding=p5)
        self.decoder_upperBlock = self._conv_block(mb_out_channels, out_channels, 
                                                    stride2=1, upsample=False, final_layer=True,
                                                    kernel_size=k3, padding=p3)
    
    @staticmethod
    def _remap_legacy_keys(state_dict, prefix, *args, **kwargs):
        for name in ("skip1", "skip2"):
            for param in ("weight", "bias"):
                old = f"{prefix}{name}.0.{param}"
                if old in state_dict:
                    state_dict[f"{prefix}{name}.{param}"] = state_dict.pop(old)

    def _conv_block(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                    stride1: int = 1, stride2: int = 2, padding: int = 1,
                    upsample: bool = False, upsample_ratio: int = 2, 
                    final_layer: bool = False) -> nn.Sequential:
        """Create a two-layer convolutional block with ELU activation."""
        first_layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride1, padding),
            nn.ELU(inplace=True)
        )
        
        if final_layer:
            second_layer = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size, stride2, padding),
            )
        else:
            second_layer = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size, stride2, padding),
                nn.ELU(inplace=True)
            )
        
        if upsample:
            return nn.Sequential(first_layer, nn.Upsample(scale_factor=upsample_ratio), second_layer)
        else:
            return nn.Sequential(first_layer, second_layer)
    
    def _conv_block_single(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                           stride: int = 1, padding: int = 1) -> nn.Sequential:
        """Create a single-layer convolutional block with LeakyReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.LeakyReLU(inplace=True)
        )
    
    def encode_features(self, x: torch.Tensor):
        """Encoder block activations (UB, MB, LB), used by the perceptual loss.

        Unlike forward hooks with ``.detach()`` (2020 code), gradients flow
        through these activations back to ``x``.
        """
        ub = self.encoder_upperBlock(x)
        mb = self.encoder_middleBlock(ub)
        lb = self.encoder_lowerBlock(mb)
        return ub, mb, lb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hourglass network.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch, out_channels * height * width)
            Flattened for compatibility with training loop.
        """
        # Encoder
        enc_out_upperBlock = self.encoder_upperBlock(x)
        enc_out_middleBlock = self.encoder_middleBlock(enc_out_upperBlock)
        enc_out_lowerBlock = self.encoder_lowerBlock(enc_out_middleBlock)
        
        # Decoder with skip connections
        dec_out_lowerBlock = self.decoder_lowerBlock(enc_out_lowerBlock)
        x = dec_out_lowerBlock + self.skip1(enc_out_middleBlock)
        
        dec_out_middleBlock = self.decoder_middleBlock(x)
        x = dec_out_middleBlock + self.skip2(enc_out_upperBlock)
        
        dec_out_upperBlock = self.decoder_upperBlock(x)
        
        # Flatten output
        x = dec_out_upperBlock.view(dec_out_upperBlock.size(0), -1)
        
        return x


class Koundinya2D_CNN(nn.Module):
    """
    Simple 2D CNN architecture for spectral reconstruction.
    
    Based on Koundinya et al. (2018) "2D-3D CNN Based Architectures for Spectral 
    Reconstruction from RGB Images".
    
    Fixed architecture: 7 input channels -> 175 output channels
    """
    
    def __init__(self):
        super(Koundinya2D_CNN, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=7, out_channels=60, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=60, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=100, out_channels=175, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input tensor of shape (batch, 7, height, width)
            
        Returns:
            Output tensor of shape (batch, 175 * height * width)
            Flattened for compatibility with training loop.
        """
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        return x
