from abc import ABC, abstractmethod


# base class for with_noise
class BaseSingleTransformation(object):

    def __init__(self, K, D):
        self.K = K
        self.D = D  # output dimension
        self.d = int(D/2)

    @abstractmethod
    def transform(self, inputs_self, inputs_other):
        pass

    @abstractmethod
    def transform_condition_on_z(self, z, inputs_self, inputs_other):
        pass

