from abc import ABC, abstractmethod

import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import check_and_convert_to_tensor

from project_ssms.unit_transformations.unit_direction_speedfree_transformation\
    import UnitDirectionSpeedFreeTransformation
from project_ssms.unit_transformations.unit_momentum_direction_transformation\
    import UnitMomentumDirectionTransformation
from project_ssms.unit_transformations.unit_direction_transformation import UnitDirectionTransformation


UNIT_TRANSFORMATION_CLASSES = dict(
    momentum_direction=UnitMomentumDirectionTransformation,
    direction=UnitDirectionTransformation,
    direction_speedfree=UnitDirectionSpeedFreeTransformation
)


class BaseWeightedDirectionTransformation(BaseTransformation):
    def __init__(self, K, D, Df, feature_vec_func, acc_factor=2, lags=1):
        super(BaseWeightedDirectionTransformation, self).__init__(K, D)
        self.d = int(self.D / 2)
        self.lags= lags

        self.Df = Df
        self.feature_vec_func = feature_vec_func
        self.acc_factor = acc_factor

    def transform(self, inputs, **kwargs):
        """
        :param inputs: (T, 4)
        :return: outputs (T, K, 4)
        """

        # calculate the weights
        T, D = inputs.shape
        assert D == self.D, "input should have last dimension = {}".format(self.D)

        # get weigths
        weights_a, weights_b = self.get_weights(inputs, **kwargs)
        #weights_a = self.get_weights(inputs[:, 0:2], 0, **kwargs)
        #weights_b = self.get_weights(inputs[:, 2:4], 1, **kwargs)
        assert weights_a.shape == (T, self.K, self.Df), \
            "weigths_a should have shape {}, instead of {}".format((T, self.K, self.Df), weights_a.shape)
        assert weights_b.shape == (T, self.K, self.Df), \
            "weights_b should have shape {}, instead of {}".format((T, self.K, self.Df), weights_b.shape)

        # feature vecs
        feature_vecs = kwargs.get("feature_vecs", None)
        if feature_vecs is None:
            #print("not using feature vecs memory")
            feature_vecs_a = self.feature_vec_func(inputs[:, 0:2])  # (T, Df, 2)
            feature_vecs_b = self.feature_vec_func(inputs[:, 2:4])  # (T, Df, 2)
        else:
            assert isinstance(feature_vecs, tuple)
            feature_vecs_a, feature_vecs_b = feature_vecs
        assert feature_vecs_a.shape == (T, self.Df, self.d), \
            "feature_vecs_a should have shape ({}, {}, {}), but is of shape {}".format(T, self.Df, self.d,
                                                                                       feature_vecs_a.shape)
        assert feature_vecs_b.shape == (T, self.Df, self.d), \
            "feature_vecs_b should have shape ({}, {}, {}), but is of shape {}".format(T, self.Df, self.d,
                                                                                       feature_vecs_b.shape)

        # make transformation
        # (T, K, Df) * (T, Df, d) --> (T, K, d)
        out_a = torch.matmul(torch.sigmoid(weights_a), feature_vecs_a)  # (T, K, 2)
        out_b = torch.matmul(torch.sigmoid(weights_b), feature_vecs_b)  # (T, K, 2)
        assert out_a.shape == (T, self.K, self.d)
        assert out_b.shape == (T, self.K, self.d)

        out = inputs[:, None, ] + self.acc_factor * torch.cat((out_a, out_b), dim=-1)  # (T, K, 4)
        assert out.shape == (T, self.K, self.D)
        return out

    def transform_condition_on_z(self, z, inputs, **kwargs):
        """

        :param z: an integer
        :param inputs: (T_pre, D)
        :return: (D, )
        """

        _, D = inputs.shape
        assert D == self.D, "input should have last dimension = {}".format(self.D)

        # get weights
        weights_a, weights_b = self.get_weights_condition_on_z(inputs, z, **kwargs)
        #weights_a = self.get_weights_condition_on_z(inputs, 0, z, **kwargs)
        #weights_b = self.get_weights_condition_on_z(inputs, 1, z, **kwargs)
        assert weights_a.shape == (1, self.Df)
        assert weights_b.shape == (1, self.Df)

        # feature vec
        feature_vec = kwargs.get("feature_vec", None)
        if feature_vec is None:
            #print("not using feature_vec memory")
            feature_vec_a = self.feature_vec_func(inputs[-1:, 0:2])
            feature_vec_b = self.feature_vec_func(inputs[-1:, 2:4])
        else:
            feature_vec_a, feature_vec_b = feature_vec

        assert feature_vec_a.shape == (1, self.Df, self.d)
        assert feature_vec_b.shape == (1, self.Df, self.d)

        # make transformation
        # (1, Df), (1, Df, d) --> (1, 1, d)
        out_a = torch.matmul(torch.sigmoid(weights_a), feature_vec_a)  # (1, 1, 2)
        out_b = torch.matmul(torch.sigmoid(weights_b), feature_vec_b)  # (1, 1, 2)
        assert out_a.shape == (1, 1, self.d)
        assert out_b.shape == (1, 1, self.d)

        out = torch.cat((out_a, out_b), dim=-1)  # (1, 1, 4)
        out = torch.squeeze(out)  # (4,)

        out = inputs[-1] + self.acc_factor * out
        assert out.shape == (self.D,)
        return out

    @abstractmethod
    def get_weights(self, inputs, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_weights_condition_on_z(self, inputs, z, **kwargs):
        raise NotImplementedError



