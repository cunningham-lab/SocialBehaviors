from abc import abstractmethod
import torch

from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.single_transformations.base_single_transformation import BaseSingleTransformation


# base class for with_noise
class SingleDirectionTransformation(BaseSingleTransformation):
    """
    x^{self}_t \sim x^{self}_{t-1} + acc_factor * [ \sum_{i=1}^{Df} sigmoid(W^k_i) f_i (self)]
    """

    def __init__(self, K, D, Df, feature_vec_func=None, acc_factor=2, dtype=torch.float32):
        super(SingleDirectionTransformation, self).__init__(K, D)
        # d = int(D/2)

        if Df is None:
            raise ValueError("Please provide number of features")
        self.Df = Df

        if feature_vec_func is None:
            raise ValueError("Must provide feature vec funcs.")
        self.feature_vec_func = feature_vec_func

        self.acc_factor = acc_factor  # int

        self.Ws = torch.rand(self.K, self.Df, dtype=dtype
                             , requires_grad=True)

    @property
    def params(self):
        return self.Ws,

    @params.setter
    def params(self, values):
        self.Ws = set_param(self.Ws, values)

    @property
    def weights(self):
        # (K, Df)
        return self.acc_factor * torch.sigmoid(self.Ws)

    @abstractmethod
    def permute(self, perm):
        self.Ws = torch.tensor(self.Ws[perm], requires_grad=True)

    @abstractmethod
    def transform(self, inputs_self, inputs_other, **memory_kwargs):
        """
        x^{self}_t \sim
        x^{self}_{t-1} + acc_factor * [ \sum_{i=1}^{Df} sigmoid(W_i) f_i (self, other)]
        :param inputs_self: (T, d)
        :param inputs_other: (T, d)
        :param momentum_vecs:
        :return: outputs_self: (T, d)
        """
        T = inputs_self.shape[0]

        feature_vecs = memory_kwargs.get("feature_vecs", None)

        if feature_vecs is None:
            feature_vecs = self.feature_vec_func(inputs_self)

        assert feature_vecs.shape == (T, self.Df, self.d), \
            "Feature vec shape is " + str(feature_vecs.shape) \
            + ". It should have shape ({}, {}, {}).".format(T, self.Df, self.d)

        # (K, Df) * (T, Df, d) -> (T, K, d)
        out = torch.matmul(torch.sigmoid(self.Ws), feature_vecs)
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
        feature_vec = memory_kwargs.get("feature_vec", None)

        if feature_vec is None:
            feature_vec = self.feature_vec_func(inputs_self[-1:])
            assert feature_vec.shape == (1, self.Df, 2)
            feature_vec = torch.squeeze(feature_vec, dim=0)
        else:
            feature_vec = check_and_convert_to_tensor(feature_vec)

        assert feature_vec.shape == (self.Df, self.d)

        # (1, Df) * (Df, d) -> (1, d)
        out = torch.matmul(torch.sigmoid(self.Ws[z][None]), feature_vec)
        assert out.shape == (1, self.d)

        out = torch.squeeze(out, dim=0)
        assert out.shape == (self.d,)

        out = inputs_self[-1] + self.acc_factor * out
        assert out.shape == (self.d,)
        return out


