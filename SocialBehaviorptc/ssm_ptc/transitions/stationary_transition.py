import torch
import numpy as np
import numpy.random as npr

from ssm_ptc.transitions.base_transition import BaseTransition
from ssm_ptc.utils import set_param


class StationaryTransition(BaseTransition):

    def __init__(self, K, D, M=0, Pi=None, **kwargs):
        super(StationaryTransition, self).__init__(K, D, M)

        if Pi is None:
            Pi = 2 * np.eye(K) + .05 * npr.rand(K, K)
        else:
            assert isinstance(Pi, np.ndarray)
            assert Pi.shape == (K, K)

        self.Pi = torch.tensor(Pi, dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return self.Pi,

    @params.setter
    def params(self, values):
        self.Pi = set_param(self.Pi, values[0])

    @property
    def stationary_transition_matrix(self):
        return torch.nn.Softmax(dim=1)(self.Pi)

    @property
    def log_stationary_transition_matrix(self):
        return torch.nn.LogSoftmax(dim=1)(self.Pi)

    def transition_matrix(self, data, input, log=False):
        if log:
            return torch.nn.LogSoftmax(dim=1)(self.Pi)

        return torch.nn.Softmax(dim=1)(self.Pi)

    def permute(self, perm):
        self.Pi = torch.tensor(self.Pi[np.ix_(perm, perm)], requires_grad=True)
