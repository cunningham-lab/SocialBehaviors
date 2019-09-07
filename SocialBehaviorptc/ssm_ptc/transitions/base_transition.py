import torch
from ssm_ptc.utils import ensure_args_are_lists_of_tensors


class BaseTransition:

    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, values):
        raise NotImplementedError

    def transition_matrix(self, data, input, log=False):
        raise NotImplementedError

    def permute(self, perm):
        raise NotImplementedError

    @ensure_args_are_lists_of_tensors
    def initialize(self, datas, inputs):
        pass

    def log_prior(self):
        return 0

