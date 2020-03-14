import torch
import torch.nn as nn
from torch.distributions import Categorical
from ssm_ptc.utils import check_and_convert_to_tensor, set_param


class BaseInitStateDistn(nn.Module):

    def __init__(self, K, D, M=0, logits=None, dtype=torch.float64):
        super(BaseInitStateDistn, self).__init__()
        self.K, self.D, self.M = K, D, M

        if logits is None:
           logits = torch.ones(self.K, dtype=dtype)
        else:
            logits = check_and_convert_to_tensor(logits, dtype=dtype)
        self.logits = nn.Parameter(logits, requires_grad=True)

    def log_prior(self):
        return 0

    @property
    def params(self):
        return self.logits,

    @params.setter
    def params(self, values):
        self.logits = set_param(self.logits, values[0])

    @property
    def log_probs(self):
        return torch.log_softmax(self.logits, dim=0)

    @property
    def probs(self):
        return torch.softmax(self.logits, dim=0)

    def permute(self, perm):
        self.logits = torch.tensor(self.logits[perm], requires_grad=True)

    def sample(self):
        m = Categorical(logits=self.logits)
        return  m.sample() # int 64


