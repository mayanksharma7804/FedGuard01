import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

from fedguard.model import build_model, count_parameters


def test_output_shape_accepts_2d_and_3d_input():
    m = build_model(38, 10).eval()
    assert m(torch.randn(7, 38)).shape == (7, 10)
    assert m(torch.randn(7, 1, 38)).shape == (7, 10)


def test_parameter_count_in_planned_range():
    n = count_parameters(build_model(38, 10))
    assert 40_000 <= n <= 60_000, n            # plan: ~40-60k (DP noise grows with sqrt(d))


def test_opacus_accepts_model_and_trains_one_dp_step():
    m = build_model(38, 10)
    assert ModuleValidator.validate(m, strict=False) == []
    loader = DataLoader(TensorDataset(torch.randn(64, 38), torch.randint(0, 10, (64,))), batch_size=16)
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    m, opt, loader = PrivacyEngine().make_private(module=m, optimizer=opt, data_loader=loader,
                                                  noise_multiplier=1.0, max_grad_norm=1.0)
    before = [p.detach().clone() for p in m.parameters()]
    for xb, yb in loader:
        if len(yb):
            opt.zero_grad(); torch.nn.functional.cross_entropy(m(xb), yb).backward(); opt.step()
            break
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_same_seed_same_initial_weights():
    torch.manual_seed(3); a = build_model(38, 10)
    torch.manual_seed(3); b = build_model(38, 10)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
