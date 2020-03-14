import torch
import numpy as np

from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.utils import random_rotation, check_and_convert_to_tensor


class ConstrainedLinearTransformation(LinearTransformation):
    """
    Actually 1-layer MLP: one linear-with_noise + scaled sigmoid
    """

    def __init__(self, K, D, bounds, lags=1, As=None, use_bias=True, bs=None, alpha=0.2):
        super(ConstrainedLinearTransformation, self).__init__(K, D, lags, As, use_bias, bs)

        self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64)
        # for each dimension in D, there should be a lower bound and an upper bound
        assert self.bounds.shape == (D, 2)

        self.alpha = alpha

    def transform(self, inputs):
        """
        Perform transformations on all possible zs
        :param inputs: (T, d_in)
        :return: (T, K, D)
        """
        out = super(ConstrainedLinearTransformation, self).transform(inputs)
        return self.scale_fn(out)

    def transform_condition_on_z(self, z, inputs):
        """
        Perform with_noise conditioning on z
        :param z: an integer
        :param inputs: (T, d_in)
        :return: (T, D)
        """
        out = super(ConstrainedLinearTransformation, self).transform_condition_on_z(z, inputs)
        return self.scale_fn(out)

    def scale_fn(self, inputs):
        """
        Scale the inputs to be centered around self.center, and to be bounded within self.bounds
        :param inputs: (T, K, D) or (T, D)
        :return: outputs: (T, K, D) or (T, D)
        """

        outputs = torch.sigmoid(inputs)
        assert outputs.shape == inputs.shape

        outputs = outputs * (self.bounds[..., 1] - self.bounds[...,0]) + self.bounds[...,0]
        assert outputs.shape == inputs.shape
        return outputs



