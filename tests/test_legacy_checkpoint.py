import torch
import torch.nn as nn

from spectral_sr.models import Hourglass


def test_loads_legacy_sequential_skip_keys():
    src = Hourglass(7, 175, 150, 100, 75)
    legacy = {}
    for k, v in src.state_dict().items():
        for name in ("skip1", "skip2"):
            if k.startswith(name + "."):
                k = k.replace(name + ".", name + ".0.", 1)
        legacy[k] = v
    dst = Hourglass(7, 175, 150, 100, 75)
    dst.load_state_dict(legacy, strict=True)
    x = torch.randn(2, 7, 64, 64)
    assert torch.allclose(src.eval()(x), dst.eval()(x))
