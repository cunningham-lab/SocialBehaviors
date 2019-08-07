import numpy as np
import torch
from ssm_ptc.utils import ensure_args_are_lists_of_tensors

class BaseObservations():

    def __init__(self, K, D, M=0):

        self.K, self.D, self.M = K, D, M

    def log_prior(self):
        return 0

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, values):
        raise NotImplementedError

    def log_prob(self, data, **kwargs):
        raise NotImplementedError

    def sample_x(self, z, xhist=None, return_np=True):
        """
        generate samples
        """

        with torch.no_grad():
            x = self.rsample_x(z, xhist)
        if return_np:
            return x.numpy()
        return x

    def rsample_x(self, z, xhist):
        raise NotImplementedError

    def permute(self, perm):
        raise NotImplementedError

    @ensure_args_are_lists_of_tensors
    def initialize(self, datas, inputs):
        pass
