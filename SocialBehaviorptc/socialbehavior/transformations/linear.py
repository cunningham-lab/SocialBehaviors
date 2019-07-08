import torch
import numpy as np

from socialbehavior.transformations.base_transformation import BaseTransformation
from socialbehavior.utils import random_rotation


class LinearTransformation(BaseTransformation):

    def __init__(self, K, d_in, d_out, As=None, use_bias=False, bs=None):
        super(LinearTransformation, self).__init__(K, d_out)

        if As is None:
            #self.As = 0.95*torch.randn(K, d_in, d_out, dtype=torch.float64, requires_grad=True)
            As = .95 * np.array([random_rotation(d_in) for _ in range(self.K)])

        self.As = torch.tensor(As, dtype=torch.float64, requires_grad=True)
        assert self.As.shape == (self.K, d_in, d_out)

        if use_bias:
            if bs is None:
                bs = np.random.rand(self.K, self.d_out)
            self.bs = torch.tensor(bs, dtype=torch.float64, requires_grad=True)
        else:
            self.bs = torch.zeros(self.K, self.d_out, dtype=torch.float64)

        assert self.bs.shape == (self.K, self.d_out)

    @property
    def params(self):
        return [self.As, self.bs]

    def permute(self, perm):
        self.As = self.As[perm]
        self.bs =self.bs[perm]

    def transform(self, inputs):
        """
        Perform transformations on all possible zs
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
        out = out + self.bs

        return out

    def transform_condition_on_z(self, z, inputs):
        """
        Perform transformation conditioning on z
        :param z: an integer
        :param inputs: (T, d_in)
        :return: (T, d_out)
        """
        A = self.As[z]

        d_in, d_out = A.shape
        T, _ = inputs.shape
        assert d_in == inputs.shape[1]

        out = torch.matmul(inputs, A) + self.bs[z]
        assert out.shape == (T, d_out)
        return out


