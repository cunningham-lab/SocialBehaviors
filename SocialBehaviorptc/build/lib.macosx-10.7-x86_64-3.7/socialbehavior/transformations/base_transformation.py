from abc import ABC, abstractmethod

# base class for transformation
class BaseTransformation(object):

    def __init__(self, d_out):
        self.d_out = d_out # output dimension

    @property
    def params(self):
        raise NotImplementedError

    @abstractmethod
    def transform(self, inputs):
        pass

