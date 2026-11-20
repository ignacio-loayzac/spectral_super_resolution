"""Hyperion band bookkeeping for the 175-band OLI/Hyperion dataset (BACKLOG T-017 / T-005).

The dataset tiles keep Hyperion bands 1..242 minus the bad/uncalibrated/water
bands below, in original band order (426.8-2365.2 nm, monotonic: the VNIR/SWIR
overlap falls inside the removed bands 58-76). Channel positions do NOT match
Hyperion band numbers, so look bands up by wavelength with :func:`band_at`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

# Same mask as legacy/utiles.py (original preprocessing): 67 bands removed -> 175 kept.
HYPERION_BAD_BANDS = sorted(
    set(range(1, 8)) | set(range(58, 77)) | set(range(121, 127))
    | set(range(167, 181)) | set(range(222, 243))
)
N_HYPERION_BANDS = 242
RETAINED_BANDS: List[int] = [b for b in range(1, N_HYPERION_BANDS + 1) if b not in HYPERION_BAD_BANDS]
assert len(RETAINED_BANDS) == 175

# Landsat-8 OLI reflective bands 1-7 (nominal centre wavelengths, nm)
OLI_CENTERS_NM = {1: 443.0, 2: 482.0, 3: 561.5, 4: 654.5, 5: 865.0, 6: 1608.5, 7: 2200.5}


class BandNotAvailable(KeyError):
    """No retained band lies within the requested tolerance."""


def parse_hyperion_wavelengths(properties_path: Union[str, Path]) -> np.ndarray:
    """Return the 242 Hyperion centre wavelengths (nm) from an ENVI properties dump."""
    regex = re.compile(r"Wavelengths (\d+) = ([\d.]+)")
    waves = {}
    with open(properties_path, "r", errors="ignore") as fh:
        for line in fh:
            m = regex.search(line)
            if m:
                waves[int(m.group(1))] = float(m.group(2))
    if len(waves) != N_HYPERION_BANDS:
        raise ValueError(f"expected {N_HYPERION_BANDS} wavelengths, found {len(waves)}")
    return np.array([waves[b] for b in range(1, N_HYPERION_BANDS + 1)])


def retained_wavelengths(properties_path: Union[str, Path]) -> np.ndarray:
    """Centre wavelengths (nm) of the 175 retained bands, in dataset channel order."""
    all_w = parse_hyperion_wavelengths(properties_path)
    return all_w[np.array(RETAINED_BANDS) - 1]


def band_at(wavelengths_nm: Sequence[float], target_nm: float, tol_nm: float = 10.0) -> int:
    """Channel index of the retained band closest to ``target_nm``.

    Raises :class:`BandNotAvailable` if the closest band is farther than ``tol_nm``
    (e.g. the target falls in a removed water-absorption window).
    """
    w = np.asarray(wavelengths_nm, dtype=float)
    idx = int(np.argmin(np.abs(w - target_nm)))
    if abs(w[idx] - target_nm) > tol_nm:
        raise BandNotAvailable(f"no band within {tol_nm} nm of {target_nm} nm (closest {w[idx]:.2f} nm)")
    return idx


def band_window(wavelengths_nm: Sequence[float], lo_nm: float, hi_nm: float) -> np.ndarray:
    """Channel indices whose centre wavelength lies in ``[lo_nm, hi_nm]``, sorted by wavelength."""
    w = np.asarray(wavelengths_nm, dtype=float)
    idx = np.where((w >= lo_nm) & (w <= hi_nm))[0]
    return idx[np.argsort(w[idx])]
