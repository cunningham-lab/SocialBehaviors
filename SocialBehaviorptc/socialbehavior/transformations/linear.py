import torch
import numpy as np

from socialbehavior.transformations.base_transformation import BaseTransformation


class LinearTransformation(BaseTransformation):

    def __init__(self, K, d_in, d_out, As=None, use_bias=False):
        super(LinearTransformation, self).__init__(d_out)

        if As is None:
            self.As = torch.randn(K, d_in, d_out, dtype=torch.float64, requires_grad=True)
        else:
            self.As = torch.tensor(As, dtype=torch.float64, requires_grad=True)
            assert self.As.shape == (K, d_in, d_out)

        #As =[np.eye(d_in, d_out), np.eye(d_in, d_out)*2, np.eye(d_in, d_out)*3]
        #self.As = torch.tensor(As, requires_grad=True)

    @property
    def params(self):
        return [self.As]

    def transform(self, inputs):
        """

        :param inputs: (T, d_in)
        :return: (T, K, d_out)
        """

        K, d_in, d_out = self.As.shape

        T, _ = inputs.shape # (T, d_in)
        assert d_in == inputs.shape[1]

        inputs = inputs.unsqueeze(0) # (1, T, d_in)

        out = torch.matmul(inputs, self.As)
        assert out.shape == (K, T, d_out)
        out = out.transpose(0,1)  # (T, K, d_out)

        return out

    def transform_condition_on_z(self, z, inputs):
        """

        :param z: an integer
        :param inputs: (T, d_in)
        :return: (T, d_out)
        """
        A = self.As[z]

        d_in, d_out = A.shape
        T, _ = inputs.shape
        assert d_in == inputs.shape[1]

        out = torch.matmul(inputs, A)
        assert out.shape == (T, d_out)
        return out


