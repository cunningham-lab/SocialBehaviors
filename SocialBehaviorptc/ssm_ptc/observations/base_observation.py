import torch
import torch.nn as nn
from ssm_ptc.utils import ensure_args_are_lists_of_tensors


class BaseObservation(nn.Module):

    def __init__(self, K, D, M=0):
        super(BaseObservation, self).__init__()
        self.K, self.D, self.M = K, D, M

    def log_prior(self):
        return 0

    def log_prob(self, data, **kwargs):
        raise NotImplementedError

    def sample_x(self, z, xhist=None, with_noise=True, return_np=True, **kwargs):
        """
        generate samples
        """

        with torch.no_grad():
            x = self.rsample_x(z, xhist, with_noise=with_noise, **kwargs)
        if return_np:
            return x.numpy()
        return x

    def rsample_x(self, z, xhist, with_noise=True, **kwargs):
        raise NotImplementedError

    @ensure_args_are_lists_of_tensors
    def initialize(self, datas, inputs):
        pass

    def log_prob(self, data, z, **kwargs):
        raise NotImplementedError
