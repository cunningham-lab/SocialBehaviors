import numpy as np


class BaseObservations():

    def __init__(self, K, D, M=0):

        self.K, self.D, self.M = K, D, M

    def log_prior(self):
        return 0

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def log_prob(self, data, input):
        raise NotImplementedError

    def sample_x(self, z, xhist):
        raise NotImplementedError

