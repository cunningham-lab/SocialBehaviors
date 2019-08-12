from abc import abstractmethod
import torch

from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.single_transformations.base_single_transformation import BaseSingleTransformation
from project_ssms.momentum_utils import get_momentum_in_batch, get_momentum


# base class for transformation
class SingleMomentumDirectionTransformation(BaseSingleTransformation):
    """
    x^{self}_t \sim x^{self}_{t-1} + acc_factor * [ sigmoid(W_0) m_t  + \sum_{i=1}^{Df} sigmoid(W_i) f_i (self, other)]
    """

    def __init__(self, K, D, Df, lags=2, momentum_weights=None,
                 feature_vec_func=None, acc_factor=2):
        super(SingleMomentumDirectionTransformation, self).__init__(K, D)
        # d = int(D/2)

        if Df is None:
            raise ValueError("Please provide number of features")
        self.Df = Df

        self.momentum_lags = lags

        if momentum_weights is None:
            self.momentum_weights = torch.ones(lags, dtype=torch.float64)
        else:
            self.momentum_weights = check_and_convert_to_tensor(momentum_weights)

        if feature_vec_func is None:
            raise ValueError("Must provide feature funcs.")
        self.feature_vec_func = feature_vec_func

        self.acc_factor = acc_factor  # int

        self.Ws = torch.rand(self.K, 1 + self.Df, dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return self.Ws,

    @params.setter
    def params(self, values):
        self.Ws = set_param(self.Ws, values)

    @abstractmethod
    def permute(self, perm):
        self.Ws = torch.tensor(self.Ws[perm], requires_grad=True)

    @abstractmethod
    def transform(self, inputs_self, inputs_other, **memory_kwargs):
        """
        x^{self}_t \sim
        x^{self}_{t-1} + acc_factor * [ sigmoid(W_0) m_t  + \sum_{i=1}^{Df} sigmoid(W_i) f_i (self, other)]
        :param inputs_self: (T, d)
        :param inputs_other: (T, d)
        :param momentum_vecs:
        :return: outputs_self: (T, d)
        """
        T = inputs_self.shape[0]

        momentum_vecs = memory_kwargs.get("momentum_vecs", None)
        feature_vecs = memory_kwargs.get("feature_vecs", None)

        if momentum_vecs is None:
            momentum_vecs = get_momentum_in_batch(inputs_self, self.momentum_lags, self.momentum_weights)
        assert momentum_vecs.shape == (T, self.d)

        if feature_vecs is None:
            feature_vecs = self.feature_vec_func(inputs_self, inputs_other)

        assert feature_vecs.shape == (T, self.Df, self.d)

        all_vecs = torch.cat((momentum_vecs[:, None, :2], feature_vecs), dim=1)  # (T, 1+Df, d)

        assert all_vecs.shape == (T, 1 + self.Df, 2)

        # (K, 1+Df) * (T, 1+Df, d) -> (T, K, d)
        out = torch.matmul(torch.sigmoid(self.Ws), all_vecs)
        assert out.shape == (T, self.K, 2)

        out = inputs_self[:, None, ] + self.acc_factor * out

        assert out.shape == (T, self.K, self.d)
        return out

    @abstractmethod
    def transform_condition_on_z(self, z, inputs_self, inputs_other, **memory_kwargs):
        """

        :param z: an integer
        :param inputs_self: (T_pre, d)
        :param inputs_other: (T_pre, d)
        :return:
        """
        momentum_vec = memory_kwargs.get("momentum_vec", None)
        feature_vec = memory_kwargs.get("feature_vec", None)

        if momentum_vec is None:
            momentum_vec = get_momentum(inputs_self, lags=self.momentum_lags,
                                          weights=self.momentum_weights)  # (d, )
        else:
            momentum_vec = check_and_convert_to_tensor(momentum_vec)
        assert momentum_vec.shape == (self.d,)

        if feature_vec is None:
            feature_vec = self.feature_vec_func(inputs_self[-1:], inputs_other[-1:])
            assert feature_vec.shape == (1, self.Df, 2)
            feature_vec = torch.squeeze(feature_vec, dim=0)
        else:
            feature_vec = check_and_convert_to_tensor(feature_vec)

        assert feature_vec.shape == (self.Df, self.d)

        all_vecs = torch.cat((momentum_vec[None, ], feature_vec), dim=0)  # (1+Df, d)

        assert all_vecs.shape == (1 + self.Df, self.d)

        # (1, 1+Df) * (1+Df, d) -> (1, d)
        out = torch.matmul(torch.sigmoid(self.Ws[z][None]), all_vecs)
        assert out.shape == (1, self.d)

        out = torch.squeeze(out, dim=0)
        assert out.shape == (self.d,)

        out = inputs_self[-1] + self.acc_factor * out
        assert out.shape == (self.d,)
        return out


