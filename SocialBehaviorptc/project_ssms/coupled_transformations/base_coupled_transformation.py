from abc import ABC, abstractmethod
import torch

from ssm_ptc.transformations.base_transformation import BaseTransformation


class BaseCoupledTransformation(BaseTransformation):
    """
    Two animals, decoupling with_noise
    """

    def __init__(self, K, D):
        super(BaseCoupledTransformation, self).__init__(K=K, D=D)
        assert D == 4

    @abstractmethod
    def transform_a(self, inputs, **memory_kwargs):
        pass

    @abstractmethod
    def transform_b(self, inputs, **memory_kwargs):
        pass

    @abstractmethod
    def transform_a_condition_on_z(self, z, inputs, **memory_kwargs):
        pass

    @abstractmethod
    def transform_b_condition_on_z(self, z, inputs, **memory_kwargs):
        pass

    def transform(self, inputs, **memory_kwargs):
        T, D = inputs.shape

        assert self.D == D

        out_a = self.transform_a(inputs, **memory_kwargs)
        out_b = self.transform_b(inputs, **memory_kwargs)

        assert out_a.shape == (T, self.K, 2)
        assert out_b.shape == (T, self.K, 2)

        outs = torch.cat((out_a, out_b))

        assert outs.shape == (T, self.K, D)

        return outs

    def transform_condition_on_z(self, z, inputs, **memory_kwargs):

        out_a = self.transform_condition_on_z(z, inputs, **memory_kwargs)
        out_b = self.transform_b_condition_on_z(z, inputs, **memory_kwargs)

        assert out_a.shape == (2, )
        assert out_b.shape == (2, )
        out = torch.cat((out_a, out_b), dim=0)
        assert out.shape == (self.D, )
        return out

