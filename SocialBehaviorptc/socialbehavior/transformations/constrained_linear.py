import torch
import numpy as np

from socialbehavior.transformations.linear import LinearTransformation
from socialbehavior.utils import random_rotation, check_and_convert_to_tensor


class ConstrainedLinearTransformation(LinearTransformation):
    """
    Actually 1-layer MLP: one linear-transformation + scaled sigmoid
    """

    def __init__(self, K, d_in, d_out, As=None, use_bias=False, bs=None, bounds=None):
        super(ConstrainedLinearTransformation, self).__init__(K, d_in, d_out, As, use_bias, bs)

        if bounds is None:
            self.bounds = None
        else:
            self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64)
            # for each dimension in d_out, there should be a lower bound and an upper bound
            assert self.bounds.shape == (d_out, 2)

            # currently consider the center as a fixed number
            self.center = torch.mean(self.bounds, dim=1)  # (d_out, )

    def transform(self, inputs):
        """
        Perform transformations on all possible zs
        :param inputs: (T, d_in)
        :return: (T, K, d_out)
        """
        out = super(ConstrainedLinearTransformation, self).transform(inputs)
        return self.scale_fn(out)

    def transform_condition_on_z(self, z, inputs):
        """
        Perform transformation conditioning on z
        :param z: an integer
        :param inputs: (T, d_in)
        :return: (T, d_out)
        """
        out = super(ConstrainedLinearTransformation, self).transform_condition_on_z(z, inputs)
        return self.scale_fn(out)

    def scale_fn(self, inputs):
        """
        Scale the inputs to be centered around self.center, and to be bounded within self.bounds
        :param inputs: (T, K, d_out) or (T, d_out)
        :return: outputs: (T, K, d_out) or (T, d_out)
        """

        outputs = torch.nn.Sigmoid()(inputs - self.center)
        assert outputs.shape == inputs.shape

        outputs = outputs * (self.bounds[:, 1] - self.bounds[:,0]) + self.bounds[:,0]
        assert outputs.shape == inputs.shape
        return outputs



