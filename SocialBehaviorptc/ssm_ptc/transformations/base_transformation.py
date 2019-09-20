from abc import ABC, abstractmethod


# base class for transformation
class BaseTransformation:

    def __init__(self, K, D):
        self.K = K
        self.D = D # output dimension

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, values):
        raise NotImplementedError

    @abstractmethod
    def transform(self, inputs, **kwargs):
        pass

    @abstractmethod
    def transform_condition_on_z(self, z, inputs, **kwargs):
        pass

    @abstractmethod
    def permute(self, perm):
        pass
