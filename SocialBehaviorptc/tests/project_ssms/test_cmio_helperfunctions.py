import torch
import numpy as np

from project_ssms.coupled_transformations.momentum_interaction_observation import normalize, compute_costheta


def test_normalize():
    v1 = torch.tensor([1., 2., 3.])
    v2 = torch.tensor([2., 3., 4.])

    v1_n = normalize(v1, norm=2)
    v2_n = normalize(v2, norm=2)

    assert torch.allclose(v1_n, torch.tensor([1./np.sqrt(14.), 2./np.sqrt(14.), 3./np.sqrt(14.)]))
    assert torch.allclose(v2_n, torch.tensor([2./np.sqrt(29.), 3./np.sqrt(29.), 4./np.sqrt(29.)]))


def test_compute_costheta():
    v1 = torch.tensor([1., 2., 3.])
    v2 = torch.tensor([2., 3., 4.])

    costheta = compute_costheta(v1, v2)

    assert torch.allclose(costheta, torch.tensor(20./np.sqrt(406.)))

    v1s = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    v2s = torch.tensor([[2., 3., 4.], [5., 6., 7.]])

    costhetas = compute_costheta(v1s, v2s)

    assert torch.allclose(costhetas, torch.tensor([20./np.sqrt(406.), 92./np.sqrt(8470.)]))


test_normalize()

test_compute_costheta()


