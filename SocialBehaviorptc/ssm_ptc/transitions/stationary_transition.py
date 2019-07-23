import torch
import numpy as np
import numpy.random as npr

from ssm_ptc.transitions.base_transition import BaseTransition
from ssm_ptc.utils import check_and_convert_to_tensor, get_np


class StationaryTransition(BaseTransition):

    def __init__(self, K, D, M=0, Pi=None):
        super(StationaryTransition, self).__init__(K, D, M)

        if Pi is None:
            Pi = 2 * np.eye(K) + .05 * npr.rand(K, K)
            self.Pi = torch.tensor(Pi, dtype=torch.float64, requires_grad=True)
        else:
            self.Pi = check_and_convert_to_tensor(Pi, dtype=torch.float64)


    @property
    def params(self):
        return (self.Pi, )

    @params.setter
    def params(self, values):
        self.Pi = torch.tensor(get_np(values[0]), dtype=self.Pi.dtype, requires_grad=self.Pi.requires_grad)

    @property
    def stationary_transition_matrix(self):
        return torch.nn.Softmax(dim=1)(self.Pi)

    def transition_matrix(self, data, input):
        return torch.nn.Softmax(dim=1)(self.Pi)

    def permute(self, perm):
        self.Pi = self.Pi[np.ix_(perm, perm)]
