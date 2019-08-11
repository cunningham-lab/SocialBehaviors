import numpy as np
import torch


class BaseDistribution:

    def __init__(self):
        pass

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, data):
        raise NotImplementedError


