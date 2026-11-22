"""LinearResidualSR starts exactly at the linear baseline and learns only the correction."""
import torch

from spectral_sr.models import LinearResidualSR


def test_starts_at_linear_solution_and_linear_is_frozen(tmp_path):
    lin = torch.nn.Conv2d(7, 175, 1)
    path = tmp_path / "lin.pt"
    torch.save(lin.state_dict(), path)
    m = LinearResidualSR(7, 175, [16, 8, 4], linear_path=str(path))
    x = torch.rand(2, 7, 8, 8)
    with torch.no_grad():
        expected = lin(x).reshape(2, -1)
    assert torch.allclose(m(x), expected, atol=1e-6)
    loss = (m(x) - torch.rand(2, 175 * 64)).pow(2).mean()
    loss.backward()
    assert all(p.grad is None for p in m.linear.parameters())
    assert m.gate.grad is not None and m.gate.grad.abs() > 0


def test_state_dict_roundtrip_without_linear_path(tmp_path):
    m = LinearResidualSR(7, 175, [16, 8, 4])
    m2 = LinearResidualSR(7, 175, [16, 8, 4], linear_path=None)
    m2.load_state_dict(m.state_dict())
    x = torch.rand(1, 7, 8, 8)
    assert torch.allclose(m(x), m2(x))
