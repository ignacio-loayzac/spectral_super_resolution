"""Dataset classes for loading spectral imagery patches."""

import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _augment_pair(*tensors: torch.Tensor):
    """Apply the same random dihedral transform (rot90 x flip, 8 options) to (C, H, W) tensors."""
    k = int(torch.randint(0, 4, (1,)))
    flip = bool(torch.randint(0, 2, (1,)))
    out = []
    for t in tensors:
        t = torch.rot90(t, k, dims=(1, 2))
        if flip:
            t = torch.flip(t, dims=(2,))
        out.append(t.contiguous())
    return out


class SpectralDataset_npy(Dataset):
    """
    Dataset for loading paired multispectral and hyperspectral image patches.
    
    Expects .npy files in separate directories for multispectral (e.g., Landsat) 
    and hyperspectral (e.g., Hyperion) images. Files should be named identically 
    in both directories to ensure correct pairing.
    
    Args:
        root_dir_multi: Path to directory containing multispectral .npy patches
        root_dir_hyp: Path to directory containing hyperspectral .npy patches
        transforms: Optional transforms to apply to both images
    """
    
    def __init__(self, 
                 root_dir_multi: str, 
                 root_dir_hyp: str,
                 transforms: Optional[object] = None,
                 scale: float = 1.0,
                 augment: bool = False):
        self.transforms = transforms
        # Values are divided by ``scale`` (10000 maps the stored reflectance x1e4 to [0, ~1])
        self.scale = float(scale)
        self.augment = augment
        
        # Use pathlib for cross-platform compatibility
        multi_path = Path(root_dir_multi)
        hyp_path = Path(root_dir_hyp)
        
        # Get sorted lists of .npy files
        self.msi_path_list = sorted(list(multi_path.glob('*.npy')))
        self.hsi_path_list = sorted(list(hyp_path.glob('*.npy')))
        
        if len(self.msi_path_list) != len(self.hsi_path_list):
            raise ValueError(
                f"Mismatch in number of files: {len(self.msi_path_list)} multispectral "
                f"vs {len(self.hsi_path_list)} hyperspectral images"
            )
        
        self.data_len = len(self.msi_path_list)
        
        if self.data_len == 0:
            raise ValueError(
                f"No .npy files found in {root_dir_multi} or {root_dir_hyp}"
            )
    
    def __getitem__(self, index: int):
        """
        Load a single paired sample.
        
        Args:
            index: Index of sample to load
            
        Returns:
            Tuple of (multispectral_image, hyperspectral_image)
            Both as tensors with shape (channels, height, width)
        """
        # Load numpy arrays and convert to tensors
        img_x = torch.from_numpy(np.load(self.msi_path_list[index])).float()
        img_y = torch.from_numpy(np.load(self.hsi_path_list[index])).float()
        
        # Permute from (H, W, C) to (C, H, W)
        img_x = img_x.permute(2, 0, 1)
        img_y = img_y.permute(2, 0, 1)
        if self.scale != 1.0:
            img_x = img_x / self.scale
            img_y = img_y / self.scale
        if self.augment:
            img_x, img_y = _augment_pair(img_x, img_y)
        
        if self.transforms is not None:
            img_x = self.transforms(img_x)
            img_y = self.transforms(img_y)
        
        return img_x, img_y
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data_len


class HyperspectralDataset_npy(Dataset):
    """
    Dataset for loading hyperspectral-only image patches.
    
    Used for training autoencoders where input and target are the same.
    
    Args:
        root_dir: Path to directory containing hyperspectral .npy patches
        transforms: Optional transforms to apply
    """
    
    def __init__(self, root_dir: str, transforms: Optional[object] = None,
                 scale: float = 1.0, augment: bool = False):
        self.transforms = transforms
        self.scale = float(scale)
        self.augment = augment
        
        # Use pathlib for cross-platform compatibility
        root_path = Path(root_dir)
        
        # Get sorted list of .npy files
        self.hsi_path_list = sorted(list(root_path.glob('*.npy')))
        self.data_len = len(self.hsi_path_list)
        
        if self.data_len == 0:
            raise ValueError(f"No .npy files found in {root_dir}")
    
    def __getitem__(self, index: int):
        """
        Load a single hyperspectral sample.
        
        Args:
            index: Index of sample to load
            
        Returns:
            Tuple of (image, image) - both the same for autoencoder training
            Each as tensor with shape (channels, height, width)
        """
        # Load numpy array and convert to tensor
        img = torch.from_numpy(np.load(self.hsi_path_list[index])).float()
        
        # Permute from (H, W, C) to (C, H, W)
        img = img.permute(2, 0, 1)
        if self.scale != 1.0:
            img = img / self.scale
        if self.augment:
            (img,) = _augment_pair(img)
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        # Return twice for autoencoder (input, target)
        return img, img
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data_len
