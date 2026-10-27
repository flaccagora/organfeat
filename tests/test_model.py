"""Descriptor model and loss: the missing-modality path equals the CT path, the zero-initialised
adapter starts from plain Sonata, and the loss rewards correct correspondences. GPU only."""

import pytest
import torch

from ofeat.models.losses import infonce

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")


def test_infonce_prefers_matching_embeddings():
    g = torch.Generator().manual_seed(0)
    p = torch.rand(500, 3, generator=g) * 100
    z = torch.nn.functional.normalize(torch.randn(500, 16, generator=g), dim=1)
    good, st = infonce(z, z, p, p, pos_radius=1e-3, safe_radius=1e-3)
    bad, _ = infonce(z, z[torch.randperm(500, generator=g)], p, p, pos_radius=1e-3, safe_radius=1e-3)
    assert st["acc"] == 1.0 and good < bad


def test_infonce_safe_radius_excludes_neighbours():
    """Two nearly identical neighbours 5 mm apart: with a 12 mm safe radius neither is a negative."""
    p = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [100.0, 0, 0]])
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1.0, 0.01], [0.0, 1.0]]), dim=1)
    loss_safe, _ = infonce(z[:1], z, p[:1], p, tau=0.1, pos_radius=1.0, safe_radius=12.0)
    loss_none, _ = infonce(z[:1], z, p[:1], p, tau=0.1, pos_radius=1.0, safe_radius=1.0)
    assert loss_safe < loss_none


@cuda
def test_missing_path_equals_ct_path_and_zero_init():
    from ofeat.mesh import data_root
    from ofeat.models.descriptor import PointDescriptor

    torch.manual_seed(0)
    m = PointDescriptor(weights_dir=str(data_root() / "weights" / "sonata")).cuda().eval()
    x = torch.randn(2000, 3, device="cuda") * 40
    n = torch.nn.functional.normalize(torch.randn(2000, 3, device="cuda"), dim=1)
    g = torch.randn(2000, 4, device="cuda")
    v = torch.randn(2000, 64, device="cuda")
    with torch.no_grad():  # PTv3 shuffles its serialization orders from the global RNG: reseed
        torch.manual_seed(1)
        a = m([x], [n], [g], [None])[0]
        torch.manual_seed(1)
        b = m([x], [n], [g], [v], [torch.zeros(2000, dtype=torch.bool, device="cuda")])[0]
        torch.manual_seed(1)
        c = m([x], [n], [g], [v])[0]
    assert torch.allclose(a, b, atol=1e-4)  # all points dropped == geometry-only path
    assert torch.allclose(a, c, atol=1e-4)  # zero-initialised adapter: images change nothing yet
    assert a.shape == (2000, 32) and torch.allclose(a.norm(dim=1), torch.ones(2000, device="cuda"), atol=1e-3)


def test_ppf_descriptor_is_rotation_invariant():
    from scipy.spatial.transform import Rotation

    from ofeat.models.local import PPFDescriptor

    torch.manual_seed(0)
    m = PPFDescriptor(radii=(10.0, 25.0), k=32).eval()
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(1500, 2, generator=g) * 120
    x = torch.cat([xy, (8 * torch.sin(xy[:, :1] / 15))], 1)
    n = torch.nn.functional.normalize(torch.cat([-8 / 15 * torch.cos(xy[:, :1] / 15), torch.zeros(1500, 1),
                                                 torch.ones(1500, 1)], 1), dim=1)
    R = torch.tensor(Rotation.random(random_state=1).as_matrix(), dtype=torch.float32)
    geo = torch.zeros(1500, 4)
    with torch.no_grad():
        a = m([x], [n], [geo])[0]
        b = m([x @ R.T + 5.0], [n @ R.T], [geo])[0]
    same = (a - b).abs().max(1).values < 1e-3  # float32 ties can reorder a neighbour subset
    assert same.float().mean() > 0.97 and a.shape == (1500, 32)
