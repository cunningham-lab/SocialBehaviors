from abc import ABC, abstractmethod


# base class for transformation
class BaseSingleTransformation(object):

    def __init__(self, K, D):
        self.K = K
        self.D = D  # output dimension
        self.d = int(D/2)

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, values):
        raise NotImplementedError

    @abstractmethod
    def permute(self, perm):
        pass

    @abstractmethod
    def transform(self, inputs_self, inputs_other):
        pass

    @abstractmethod
    def transform_condition_on_z(self, z, inputs_self, inputs_other):
        pass

