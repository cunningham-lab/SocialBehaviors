import torch
import torch.nn as nn
from ssm_ptc.utils import ensure_args_are_lists_of_tensors


class BaseTransition(nn.Module):

    def __init__(self, K, D, M=0):
        super(BaseTransition, self).__init__()
        self.K, self.D, self.M = K, D, M

    def transition_matrix(self, data, input, log=False):
        raise NotImplementedError

    @ensure_args_are_lists_of_tensors
    def initialize(self, datas, inputs):
        pass

    def log_prior(self):
        return 0

