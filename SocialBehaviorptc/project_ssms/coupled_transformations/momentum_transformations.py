from ssm_ptc.observations.base_observation import BaseObservation
from ssm_ptc.distributions.truncatednormal import TruncatedNormal
from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.coupled_transformations.base_coupled_transformation import BaseCoupledTransformation
from project_ssms.momentum_utils import get_momentum_in_batch, get_momentum


import torch
import numpy as np


def normalize(f, norm=1):
    # normalize across last dimension
    if norm==1:
        f = f / torch.sum(f, dim=-1, keepdim=True)
    elif norm == 2:
        f = f / torch.norm(f, dim=-1, keepdim=True)
    return f


class DirectionTransformation(BaseCoupledTransformation):
    """
    transformation:
    x^a_t \sim x^a_{t-1} + acc_factor * [ sigmoid(W^a_0) m_t  + \sum_{i=1}^{Df} sigmoid(W^a_i) f_i ]
    x^b_t \sim x^b_{t-1} + acc_factor * [ sigmoid(W^b_0) m_t  + \sum_{i=1}^{Df} sigmoid(W^b_i) f_i ]


    feature computation
    training mode: receive precomputed feature input
    sampling mode: compute the feature based on previous observation
    """
    def __init__(self, K, D=4, Df=None, momentum_lags=2, momentum_weights=None,
                 feature_vec_func=None, acc_factor=2):
        """

        :param K: number of hidden states
        :param D: dimension of observations
        :param Df: number of direction vectors (not including the momentum vectors)
        :param momentum_lags: number of time lags to accumulate the momentum
        :param momentum_weights: weights for weighted linear regression
        :param feature_vec_func: function to compute the featured direction vector --> (T, 2*Df)
        :param acc_factor: acceleration factor, for the purpose of speed control
        """
        super(DirectionTransformation, self).__init__(K, D)
        assert D == 4

        if Df is None:
            raise ValueError("Please provide number of features")
        self.Df = Df

        self.momentum_lags = momentum_lags

        if momentum_weights is None:
            self.momentum_weights = torch.ones(momentum_lags, dtype=torch.float64)
        else:
            self.momentum_weights = check_and_convert_to_tensor(momentum_weights)

        if feature_vec_func is None:
            raise ValueError("Must provide feature funcs.")
        self.feature_vec_func = feature_vec_func

        self.acc_factor = acc_factor  # int

        self.Ws = torch.rand(self.K, 2, 1 + self.Df, dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return self.Ws,

    @params.setter
    def params(self, values):
        self.Ws = set_param(self.Ws, values[0])

    def permute(self, perm):
        self.Ws = self.Ws[perm]

    def transform_a(self, inputs, **memory_kwargs):
        T, _ = inputs.shape

        momentum_vecs_a = memory_kwargs.get("momentum_vecs_a", None)
        feature_vecs_a = memory_kwargs.get("feature_vecs_a", None)

        if momentum_vecs_a is None:
            momentum_vecs_a = get_momentum_in_batch(inputs[:, 0:2],
                                               lags=self.momentum_lags, weights=self.momentum_weights)  # (T, 2)
        assert momentum_vecs_a.shape == (T, 2)

        if feature_vecs_a is None:
            feature_vecs_a = self.feature_vec_func(inputs[..., :2], inputs[..., 2:])
        assert feature_vecs_a.shape == (T, self.Df, 2)

        all_vecs_a = torch.cat((momentum_vecs_a[:, None], feature_vecs_a), dim=1)  # (T, 1+Df, 2)

        # (K, 1+Df) * (T, 1+Df, 2) -> (T, K, 2)
        out_a = inputs[:, :2] + self.acc_factor * torch.matmul(self.Ws[:, 0], all_vecs_a)

        return out_a

    def transform_b(self, inputs, **memory_kwargs):
        T, _ = inputs.shape

        momentum_vecs_b = memory_kwargs.get("momentum_vecs_b", None)
        feature_vecs_b = memory_kwargs.get("feature_vecs_b", None)

        if momentum_vecs_b is None:
            momentum_vecs_b = get_momentum_in_batch(inputs[:, 2:4],
                                                lags=self.momentum_lags, weights=self.momentum_weights)  # (T, 2)
        assert momentum_vecs_b.shape == (T, 2)

        if feature_vecs_b is None:
            feature_vecs_b = self.feature_vec_func(inputs[..., 2:], inputs[..., :2])
        assert feature_vecs_b.shape == (T, self.Df, 2)

        all_vecs_b = torch.cat((momentum_vecs_b[:, None], feature_vecs_b), dim=1)  # (T, 1+Df, 2)

        # (K, 1+Df) * (T, 1+Df, 2) -> (T, K, 2)
        out_b = inputs[:, 2:] + self.acc_factor * torch.matmul(self.Ws[:, 0], all_vecs_b)

        return out_b

    def transform_a_condition_on_z(self, z, inputs, **memory_kwargs):
        momentum_vec_a = memory_kwargs.get("momentum_vecs_a", None)
        feature_vec_a = memory_kwargs.get("feature_vecs_a", None)

        if momentum_vec_a is None:
            momentum_vec_a = get_momentum(inputs[:, 0:2], lags=self.momentum_lags,
                                          weights=self.momentum_weights)  # (2, )
        assert momentum_vec_a.shape == (2, )

        if feature_vec_a is None:
            feature_vec_a = self.feature_vec_func(inputs[..., :2], inputs[..., 2:])
            assert feature_vec_a.shape == (1, self.Df, 2)
            feature_vec_a = torch.squeeze(feature_vec_a, dim=0)
        assert feature_vec_a.shape == (self.Df, 2)

        all_vecs_a = torch.cat((momentum_vec_a[None, ], feature_vec_a), dim=0)  # (1+Df, 2)

        # (1, 1+Df) * (1+Df, 2) -> (1, 2)
        out_a = inputs[-1, :2] + self.acc_factor * torch.matmul(self.Ws[z, 0:1], all_vecs_a)
        assert out_a.shape == (1, 2)

        return out_a

    def transform_b_condition_on_z(self, z, inputs, **memory_kwargs):
        momentum_vec_b = memory_kwargs.get("momentum_vecs_b", None)
        feature_vec_b = memory_kwargs.get("feature_vecs_b", None)

        if momentum_vec_b is None:
            momentum_vec_b = get_momentum(inputs[:, 2:4], lags=self.momentum_lags,
                                          weights=self.momentum_weights)  # (2, )
        assert momentum_vec_b.shape == (2,)

        if feature_vec_b is None:
            feature_vec_b = self.feature_vec_func(inputs[..., 2:], inputs[..., :2])
            assert feature_vec_b.shape == (1, self.Df, 2)
            feature_vec_b = torch.squeeze(feature_vec_b, dim=0)
        assert feature_vec_b.shape == (self.Df, 2)

        all_vecs_b = torch.cat((momentum_vec_b[None,], feature_vec_b), dim=0)  # (1+Df, 2)

        # (1, 1+Df) * (1+Df, 2) -> (1, 2)
        out_b = inputs[-1, 2:] + self.acc_factor * torch.matmul(self.Ws[z, 0:1], all_vecs_b)
        assert out_b.shape == (1, 2)

        return out_b


