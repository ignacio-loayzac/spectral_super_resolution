import numpy as np
import pytest

from spectral_sr import bands


@pytest.fixture
def props(tmp_path):
    # Hyperion-like grid: VNIR 1-70 from 355.59 nm step 10.18, SWIR 71-242 from 851.92 nm step 10.09
    lines = ["Wavelengths Units = nm"]
    for b in range(1, 243):
        w = 355.59 + 10.18 * (b - 1) if b <= 70 else 851.92 + 10.09 * (b - 71)
        lines.append(f"Wavelengths {b} = {w:.2f}")
    p = tmp_path / "props.txt"
    p.write_text("\n".join(lines))
    return p


def test_175_retained(props):
    w = bands.retained_wavelengths(props)
    assert w.shape == (175,)
    assert len(bands.HYPERION_BAD_BANDS) == 67


def test_band_at_and_window(props):
    w = bands.retained_wavelengths(props)
    i = bands.band_at(w, 670.0)
    assert abs(w[i] - 670.0) <= 10.0
    with pytest.raises(bands.BandNotAvailable):
        bands.band_at(w, 1400.0, tol_nm=5.0)  # inside removed water band 121-126
    win = bands.band_window(w, 2100, 2250)
    assert np.all(np.diff(w[win]) > 0) and w[win].min() >= 2100
