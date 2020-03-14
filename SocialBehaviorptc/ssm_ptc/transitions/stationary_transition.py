import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from ssm_ptc.transitions.base_transition import BaseTransition
from ssm_ptc.utils import set_param


class StationaryTransition(BaseTransition):

    def __init__(self, K, D, M=0, logits=None, dtype=torch.float64, **kwargs):
        super(StationaryTransition, self).__init__(K, D, M)
        if logits is None:
            logits = 2 * np.eye(self.K) + .05 * npr.rand(self.K, self.K)
        else:
            assert isinstance(logits, np.ndarray)
            assert logits.shape == (self.K, self.K)
        self.logits = nn.Parameter(torch.tensor(logits, dtype=dtype), requires_grad=True)

    @property
    def params(self):
        return self.logits,

    @params.setter
    def params(self, values):
        self.logits = set_param(self.logits, values[0])

    @property
    def stationary_transition_matrix(self):
        return torch.nn.Softmax(dim=-1)(self.logits)

    @property
    def log_stationary_transition_matrix(self):
        return torch.nn.LogSoftmax(dim=-1)(self.logits)

    def transition_matrix(self, data, input, log=False):
        if log:
            return torch.nn.LogSoftmax(dim=-1)(self.logits)

        return torch.nn.Softmax(dim=-1)(self.logits)

    def permute(self, perm):
        self.logits = torch.tensor(self.logits.detach().numpy()[np.ix_(perm, perm)], requires_grad=True)

# TODO: fix tomorrow
class ConstrainedStationaryTransition(BaseTransition):
    def __init__(self, K, D, M=0, logits_=None, dtype=torch.float64, **kwargs):
        super(ConstrainedStationaryTransition, self).__init__(K, D, M)
        if logits_ is None:
            logits_ = npr.rand(self.K, self.K-1)
        else:
            assert logits_.shape == (self.K, self.K-1), logits_.shape
        self.logits_ = nn.Parameter(torch.tensor(logits_, dtype=dtype), requires_grad=True)

    @property
    def params(self):
        return self.logits_,

    @params.setter
    def params(self, values):
        self.logits_ = values[0]

    @property
    def stationary_transition_matrix(self):
        probs_ = torch.softmax(self.logits_, dim=-1)  # (K, K-1)
        probs = probs_.new_zeros((self.K, self.K))
        for k in range(self.K):
            probs[k, 0:k] = probs_[0:k]
            probs[k, k+1:] = probs[k:]
        return probs

    @property
    def log_stationary_transition_matrix(self):
        log_probs_ = torch.log_softmax(self.logits_, dim=-1)  # (K, K-1)
        log_probs = log_probs_.new_empty((self.K, self.K)).fill_(-np.inf)
        for k in range(self.K):
            log_probs[k, 0:k] = log_probs_[k, 0:k]
            log_probs[k, k+1:] = log_probs_[k, k:]
        return log_probs






