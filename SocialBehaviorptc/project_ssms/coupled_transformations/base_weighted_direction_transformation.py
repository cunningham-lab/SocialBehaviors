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

        # get weights (apply scaling and sigmoid here)
        weights_a, weights_b = self.get_weights(inputs, **kwargs)
        assert weights_a.shape == (T, self.K, self.Df), \
            "weigths_a should have shape {}, instead of {}".format((T, self.K, self.Df), weights_a.shape)
        assert weights_b.shape == (T, self.K, self.Df), \
            "weights_b should have shape {}, instead of {}".format((T, self.K, self.Df), weights_b.shape)

        # feature vecs
        # TODO: accomodate to the subclasses
        feature_vecs_a = kwargs.get("feature_vecs_a", None)
        if feature_vecs_a is None:
            #print("not using feature_vecs_a memories")
            feature_vecs_a = self.feature_vec_func(inputs[:, 0:2])
        assert feature_vecs_a.shape == (T, self.Df, 2)
        feature_vecs_b = kwargs.get("feature_vecs_b", None)
        if feature_vecs_b is None:
            #print("not using feature_vecs_b memories")
            feature_vecs_b = self.feature_vec_func(inputs[:,2:4])
        assert feature_vecs_b.shape == (T, self.Df, 2)

        # make transformation
        # (T, K, Df) * (T, Df, d) --> (T, K, d)
        out_a = torch.matmul(weights_a, feature_vecs_a)
        out_b = torch.matmul(weights_b, feature_vecs_b)
        assert out_a.shape == (T, self.K, self.d)
        assert out_b.shape == (T, self.K, self.d)

        out = inputs[:, None, ] + torch.cat((out_a, out_b), dim=-1)  # (T, K, 4)
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

        # get weights (apply scaling and sigmoid here)
        weights_a, weights_b = self.get_weights_condition_on_z(inputs, z, **kwargs)
        assert weights_a.shape == (1, self.Df)
        assert weights_b.shape == (1, self.Df)

        # feature vec
        feature_vec_a = kwargs.get("feature_vec_a", None)
        feature_vec_b = kwargs.get("feature_vec_b", None)
        if feature_vec_a is None:
            #print("not using feature_vec memory")
            feature_vec_a = self.feature_vec_func(inputs[-1:, 0:2])

        if feature_vec_b is None:
            feature_vec_b = self.feature_vec_func(inputs[-1:, 2:4])

        assert feature_vec_a.shape == (1, self.Df, self.d)
        assert feature_vec_b.shape == (1, self.Df, self.d)

        # make transformation
        # (1, Df), (1, Df, d) --> (1, 1, d)
        out_a = torch.matmul(weights_a, feature_vec_a)
        out_b = torch.matmul(weights_b, feature_vec_b)
        assert out_a.shape == (1, 1, self.d)
        assert out_b.shape == (1, 1, self.d)

        out = torch.cat((out_a, out_b), dim=-1)  # (1, 1, 4)
        out = torch.squeeze(out)  # (4,)

        out = inputs[-1] + out
        assert out.shape == (self.D,)
        return out

    @abstractmethod
    def get_weights(self, inputs, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_weights_condition_on_z(self, inputs, z, **kwargs):
        raise NotImplementedError



