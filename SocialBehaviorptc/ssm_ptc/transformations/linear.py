import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import random_rotation


class LinearTransformation(BaseTransformation):

    def __init__(self, K, d_in, d_out, As=None, use_bias=False, bs=None):
        super(LinearTransformation, self).__init__(K, d_out)

        self.lags = int(d_in / d_out)

        if As is None:
            # self.As = 0.95*torch.randn(K, d_in, d_out, dtype=torch.float64, requires_grad=True)
            #As = .95 * np.array([random_rotation(d_in) for _ in range(self.K)])
            As = .95 * np.array([
                np.column_stack([random_rotation(self.d_out), np.zeros((self.d_out, (self.lags-1) * self.d_out))])
            for _ in range(K)])

        self.As = torch.tensor(As, dtype=torch.float64, requires_grad=True)
        assert self.As.shape == (self.K, d_out, d_out * self.lags)

        if use_bias:
            if bs is None:
                bs = np.random.randn(self.K, self.d_out)
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
        :param inputs: (T, d_out)
        :return: (T-lags, K, d_out)
        """
        # TODO: test lags

        K, d_in, d_out = self.As.shape

        T, _ = inputs.shape # (T, d_in)

        out = 0
        for l in range(self.lags):
            Als = self.As[:, l* d_out : (l+1) * d_out,:]  # (K, D, D)
            lagged_data = inputs[l: T-self.lags+l]
            assert lagged_data.shape == (T-self.lags, d_out)
            lagged_data = lagged_data.unsqueeze(0) # (1, T-lags, D)
            out = out + torch.matmul(lagged_data, Als)
            assert out.shape == (K, T-self.lags, d_out)

        out = out.transpose(0,1)  # (T-lags, K, d_out)
        out = out + self.bs

        return out

    def transform_condition_on_z(self, z, inputs):
        """
        Perform transformation conditioning on z
        :param z: an integer
        :param inputs: (T, d_in)
        :return: (T-lags, d_out)
        """
        # TODO: test lags
        A = self.As[z]  # (d_in, d_out)

        d_in, d_out = A.shape
        T, _ = inputs.shape

        out = 0
        for l in range(self.lags):
            Al = A[l*d_out : (l+1)*d_out]  # (D, D)
            lagged_data = inputs[l: T-self.lags+l]  # (T-lags, D)
            out = out + torch.matmul(lagged_data, Al)  # (T-lags, D)

        out = out + self.bs[z]
        assert out.shape == (T, d_out)
        return out


