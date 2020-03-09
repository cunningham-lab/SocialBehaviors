import numpy as np
import torch
from ssm_ptc.utils import check_and_convert_to_tensor, set_param


class BaseInitStateDistn:

    def __init__(self, K, D, M=0, pi=None, dtype=torch.float64, device=None):

        self.K, self.D, self.M = K, D, M

        if pi is None:
            self.pi = torch.ones(self.K, dtype=dtype, requires_grad=True, device=device)
        else:
            self.pi = check_and_convert_to_tensor(pi, dtype=dtype, requires_grad=True, device=device)

    def log_prior(self):
        return 0

    @property
    def params(self):
        return self.pi,

    @params.setter
    def params(self, values):
        self.pi = set_param(self.pi0, values[0])

    @property
    def log_pi(self):
        return torch.softmax(self.pi, dim=0) # (K, z0

    def permute(self, perm):
        self.pi = torch.tensor(self.pi[perm], requires_grad=True)

