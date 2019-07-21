import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import random_rotation


class LinearTransformation(BaseTransformation):

    def __init__(self, K, D, lags=1, As=None, use_bias=True, bs=None):
        super(LinearTransformation, self).__init__(K, D)

        self.lags = lags

        if As is None:
            As = .95 * np.array([
                np.column_stack([random_rotation(self.D), np.zeros((self.D, (self.lags - 1) * self.D))])
            for _ in range(K)])

        self.As = torch.tensor(As, dtype=torch.float64, requires_grad=True)
        assert self.As.shape == (self.K, D, D * self.lags)

        if use_bias:
            if bs is None:
                bs = np.random.randn(self.K, self.D)
            self.bs = torch.tensor(bs, dtype=torch.float64, requires_grad=True)
        else:
            self.bs = torch.zeros(self.K, self.D, dtype=torch.float64)

        assert self.bs.shape == (self.K, self.D)

    @property
    def params(self):
        return [self.As, self.bs]

    def permute(self, perm):
        self.As = self.As[perm]
        self.bs =self.bs[perm]

    def transform(self, inputs):
        """
        Perform transformations on all possible zs
        :param inputs: (T, D)
        :return: (T-lags+1, K, D)
        """

        K, D, _ = self.As.shape

        T, _ = inputs.shape # (T, D)

        out = 0
        for l in range(self.lags):
            Als = self.As[:, :, l* D : (l+1) * D]  # (K, D, D)
            lagged_data = inputs[l: T-self.lags+1+l]  # T-lags
            assert lagged_data.shape == (T-self.lags+1, D)
            lagged_data = lagged_data.unsqueeze(0) # (1, T-lags+1, D)
            out = out + torch.matmul(lagged_data, Als)
            assert out.shape == (K, T-self.lags+1, D)

        out = out.transpose(0,1)  # (T-lags+1, K, D)
        out = out + self.bs

        return out

    def transform_condition_on_z(self, z, inputs):
        """
        Perform transformation conditioning on z,
        :param z: an integer
        :param inputs: (lags, D)
        :return: (D, )
        """

        A = self.As[z]  # (D, D * lags)

        D, _ = A.shape
        T, _ = inputs.shape

        out = 0
        for l in range(self.lags):
            Al = A[:,l*D : (l+1)*D]  # (D, D)
            lagged_data = inputs[l]  # (D,)
            assert lagged_data.shape == (D, )
            out = out + torch.matmul(lagged_data, Al)  # (D, )

        out = out + self.bs[z]
        assert out.shape == (D, )
        return out


