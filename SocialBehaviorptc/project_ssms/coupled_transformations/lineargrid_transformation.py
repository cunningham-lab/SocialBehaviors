import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import set_param, check_and_convert_to_tensor


class LinearGridTransformation(BaseTransformation):
    """
    Learnable parameters: weights of each grid vertex
    weights of any random point = 2D linear interpolation of the nearby four grid points
    """

    def __init__(self, K, D, x_grids, y_grids, Df, feature_vec_func, acc_factor=2):
        super(LinearGridTransformation, self).__init__(K, D)

        self.d = int(self.D / 2)

        self.x_grids = check_and_convert_to_tensor(x_grids, dtype=torch.float64)  # [x_0, x_1, ..., x_m]
        self.y_grids = check_and_convert_to_tensor(y_grids, dtype=torch.float64)  # a list [y_0, y_1, ..., y_n]

        self.Df = Df
        self.feature_vec_func = feature_vec_func
        self.acc_factor = acc_factor

        self.gridpoints = torch.tensor([(x_grid, y_grid) for x_grid in self.x_grids for y_grid in self.y_grids])
        # number of basis grid points
        self.GP = self.gridpoints.shape[0]
        self.Ws = torch.rand(self.K, self.GP, self.Df, dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return self.Ws,

    @params.setter
    def params(self, values):
        self.Ws = set_param(self.Ws, values)

    def permute(self, perm):
        self.Ws = self.Ws[perm]

    def transform(self, inputs, memory_kwargs=None):
        # do a 2D interpolation of weights, and use that for transformation

        # first, find the grid basis and then compute the weights

        # finally, compute the transformation based on weights and positions (directions)

        T, D = inputs.shape
        assert D == self.D, "input should have last dimension = {}".format(self.D)

        weights_a = self.get_weights(inputs[:, 0:2])
        weights_b = self.get_weights(inputs[:, 2:4])
        assert weights_a.shape == (T, self.K, self.Df)
        assert weights_b.shape == (T, self.K, self.Df)

        memory_kwargs = memory_kwargs or {}
        feature_vecs = memory_kwargs.get("feature_vecs", None)
        if feature_vecs is None:
            feature_vecs_a = self.feature_vec_func(inputs[:, 0:2])  # (T, Df, 2)
            feature_vecs_b = self.feature_vec_func(inputs[:, 2:4])  # (T, Df, 2)
        else:
            feature_vecs_a, feature_vecs_b = feature_vecs
        assert feature_vecs_a.shape == (T, self.Df, self.d)
        assert feature_vecs_b.shape == (T, self.Df, self.d)

        out_a = torch.matmul(weights_a, feature_vecs_a)  # (T, K, 2)
        out_b = torch.matmul(weights_b, feature_vecs_b)  # (T, K, 2)
        assert out_a.shape == (T, self.K, self.d)
        assert out_b.shape == (T, self.K, self.d)

        out = inputs[:, None, ] + self.acc_factor * torch.cat((out_a, out_b), dim=-1)  # (T, K, 4)
        assert out.shape == (T, self.K, self.D)
        return out

    def transform_condition_on_z(self, z, inputs, memory_kwargs=None):
        """

        :param z: an integer
        :param inputs: (T_pre, D)
        :param memory_kwargs:
        :return:
        """

        _, D = inputs.shape
        assert D == self.D, "input should have last dimension = {}".format(self.D)

        memory_kwargs = memory_kwargs or {}
        feature_vec = memory_kwargs.get("feature_vec", None)

        if feature_vec is None:
            feature_vec_a = self.feature_vec_func(inputs[-1:, 0:2])
            feature_vec_b = self.feature_vec_func(inputs[-1:, 2:4])
        else:
            feature_vec_a, feature_vec_b = feature_vec

        assert feature_vec_a.shape == (1, self.Df, self.d)
        assert feature_vec_b.shape == (1, self.Df, self.d)

        weights_a = self.get_weights_for_single_point(inputs[-1, 0:2], z)  # (1, 1, Df)
        weights_b = self.get_weights_for_single_point(inputs[-1, 2:4], z)  # (1, 1, Df)
        assert weights_a.shape == (1, 1, self.Df)
        assert weights_b.shape == (1, 1, self.Df)

        out_a = torch.matmul(weights_a, feature_vec_a)  # (1, 1, 2)
        out_b = torch.matmul(weights_b, feature_vec_b)  # (1, 1, 2)
        assert out_a.shape == (1, 1, self.d)
        assert out_b.shape == (1, 1, self.d)

        out = torch.cat((out_a, out_b), dim=-1)  # (1, 1, 4)
        out = torch.squeeze(out)  # (4,)

        out = inputs[-1] + self.acc_factor * out
        assert out.shape == (self.D, )
        return out

    @staticmethod
    def get_grid_point_idx(point, x_grids, y_grids):
        find = False

        l_y = len(y_grids)
        for i in range(len(x_grids)-1):
            for j in range(len(y_grids)-1):
                cond_x = x_grids[i] <= point[0] <= x_grids[i+1]
                cond_y = y_grids[j] <= point[1] <= y_grids[j+1]
                if cond_x & cond_y:
                    find = True
                    idx = [i*l_y+j, i*l_y+j+1, (i+1)*l_y+j, (i+1)*l_y+j+1]  # (Q11, Q12, Q21, Q22)
                    break
            if find:
                break
        if not find:
            raise ValueError("value {} out of the grid world.".format(point.numpy()))
        return idx

    def get_basis_point_and_weight(self, point, z=None):
        """

        :param point: (2, )
        :param z: a scalar
        :return:
        """
        grid_points_idx = self.get_grid_point_idx(point, self.x_grids, self.y_grids)

        Qs = [self.gridpoints[grid_points_idx[0]], self.gridpoints[grid_points_idx[-1]]]  # (Q11, Q22), each is (2,)
        if z is not None:
            ws = [self.Ws[z, i] for i in grid_points_idx]  # w_Q11, w_Q12, w_Q21, w_Q22, each is (Df, )
            assert ws[0].shape == (self.Df, )
        else:
            ws = [self.Ws[:, i] for i in grid_points_idx]  # w_Q11, w_Q12, w_Q21, w_Q22, each is (K, Df)
            assert ws[0].shape == (self.K, self.Df)

        return Qs + ws

    def get_weights_for_single_point(self, point, z):
        out = self.get_basis_point_and_weight(point, z)

        Q11 = out[0]  # (2,)
        Q22 = out[1]  # (2,)

        w_Q11 = out[2]  # (Df,)
        w_Q12 = out[3]  # (Df, )
        w_Q21 = out[4]  # (Df,)
        w_Q22 = out[5]  # (Df,)

        weight = two_d_interpolation(point[None, ], Q11[None,], Q22[None,],
                                     w_Q11[None, None, ], w_Q12[None, None, ], w_Q21[None, None, ], w_Q22[None, None, ])
        assert weight.shape == (1, 1, self.Df)
        return weight

    def get_weights(self, points):
        """

        :param points: (T, 2)
        :return: (T, Df)
        """
        T, d = points.shape
        assert d == self.d

        out = list(map(self.get_basis_point_and_weight, points))

        Q11 = torch.stack([item[0] for item in out], dim=0)  # (T, 2)
        Q22 = torch.stack([item[1] for item in out], dim=0)  # (T, 2)
        assert Q11.shape == (T, 2)
        assert Q22.shape == (T, 2)

        w_Q11 = torch.stack([item[2] for item in out], dim=0)  # (T, K, Df)
        w_Q12 = torch.stack([item[3] for item in out], dim=0)
        w_Q21 = torch.stack([item[4] for item in out], dim=0)
        w_Q22 = torch.stack([item[5] for item in out], dim=0)
        assert w_Q11.shape == (T, self.K, self.Df)

        weights = two_d_interpolation(points, Q11, Q22, w_Q11, w_Q12, w_Q21, w_Q22)
        return weights


def one_d_interpolation(x, x1, x2, f_x1, f_x2):
    """

    :param x: (T, 1)
    :param x1: (T, 1)
    :param x2: (T, 1)
    :param f_x1: (T, K, Df)
    :param f_x2: (T, K, Df)
    :return: (T, K, Df)
    """

    factor_1 = (x2 - x) / (x2 - x1) # (T, 1)
    factor_2 = (x - x1) / (x2 - x1) # (T, 1)

    return factor_1[..., None] * f_x1 + factor_2[..., None] * f_x2


def two_d_interpolation(points, Q11, Q22, f_Q11, f_Q12, f_Q21, f_Q22):


    T, K, Df = f_Q11.shape

    x, y = points[:, 0:1], points[:, 1:2]  # each is (T, 1)
    x1, y1 = Q11[:, 0:1], Q11[:, 1:2]  # each is (T, 1)
    x2, y2 = Q22[:, 0:1], Q22[:, 1:2]  # each is (T, 1)

    f_Qxy1 = one_d_interpolation(x, x1, x2, f_Q11, f_Q21)  # (T, K, Df)
    f_Qxy2 = one_d_interpolation(x, x1, x2, f_Q12, f_Q22)  # (T, K, Df)
    out = one_d_interpolation(y, y1, y2, f_Qxy1, f_Qxy2)  # (T, K, Df)
    assert out.shape == (T, K, Df)
    return out


