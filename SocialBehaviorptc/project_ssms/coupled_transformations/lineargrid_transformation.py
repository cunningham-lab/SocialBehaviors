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

        # shape: (d, GP)
        self.gridpoints = torch.tensor([(x_grid, y_grid) for x_grid in self.x_grids for y_grid in self.y_grids])
        self.gridpoints = torch.transpose(self.gridpoints, 0, 1)

        # number of basis grid points
        self.GP = self.gridpoints.shape[1]
        self.Ws = torch.rand(self.K, 2, self.GP, self.Df, dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return self.Ws,

    @params.setter
    def params(self, values):
        self.Ws = set_param(self.Ws, values)

    def permute(self, perm):
        self.Ws = self.Ws[perm]

    def transform(self, inputs, gridpoints=None, gridpoints_idx=None, feature_vecs=None):
        # do a 2D interpolation of weights, and use that for transformation

        # first, find the grid basis and then compute the weights

        # finally, compute the transformation based on weights and positions (directions)

        T, D = inputs.shape
        assert D == self.D, "input should have last dimension = {}".format(self.D)

        if gridpoints_idx is None:
            # one hot vectors
            gridpoints_idx_a = self.get_gridpoints_idx_for_batch(inputs[:, 0:2])
            gridpoints_idx_b = self.get_gridpoints_idx_for_batch(inputs[:, 2:4])
        else:
            assert isinstance(gridpoints_idx, tuple)
            gridpoints_idx_a, gridpoints_idx_b = gridpoints_idx
            #print("Using grid points idx memory!")
        assert gridpoints_idx_a.shape == (T, self.GP, 4), "gridpoints_idx_a.shape = " + str(gridpoints_idx_a.shape) \
                                                          + ". The correct shape is ({}, {}, {})".format(T, self.GP, 4)
        assert gridpoints_idx_b.shape == (T, self.GP, 4), "gridpoints_idx_b.shape = " + str(gridpoints_idx_b.shape) \
                                                          + ". The correct shape is ({}, {}, {})".format(T, self.GP, 4)

        if gridpoints is None:
            gridpoints_a = self.get_gridpoints_for_batch(gridpoints_idx_a)
            gridpoints_b = self.get_gridpoints_for_batch(gridpoints_idx_b)
        else:
            assert isinstance(gridpoints, tuple)
            gridpoints_a, gridpoints_b = gridpoints
            #print("Using grid points memory!")
        assert gridpoints_a.shape == (T, self.d, 2)
        assert gridpoints_b.shape == (T, self.d, 2)

        weights_a = self.get_weights_for_batch(inputs[:, 0:2], 0, gridpoints_a, gridpoints_idx_a)  # (T, K, Df)
        weights_b = self.get_weights_for_batch(inputs[:, 2:4], 1, gridpoints_b, gridpoints_idx_b)

        if feature_vecs is None:
            feature_vecs_a = self.feature_vec_func(inputs[:, 0:2])  # (T, Df, 2)
            feature_vecs_b = self.feature_vec_func(inputs[:, 2:4])  # (T, Df, 2)
        else:
            assert isinstance(feature_vecs, tuple)
            feature_vecs_a, feature_vecs_b = feature_vecs
            #print("Using feature vec memory!")
        assert feature_vecs_a.shape == (T, self.Df, self.d)
        assert feature_vecs_b.shape == (T, self.Df, self.d)

        out_a = torch.matmul(torch.sigmoid(weights_a), feature_vecs_a)  # (T, K, 2)
        out_b = torch.matmul(torch.sigmoid(weights_b), feature_vecs_b)  # (T, K, 2)
        assert out_a.shape == (T, self.K, self.d)
        assert out_b.shape == (T, self.K, self.d)

        out = inputs[:, None, ] + self.acc_factor * torch.cat((out_a, out_b), dim=-1)  # (T, K, 4)
        assert out.shape == (T, self.K, self.D)
        return out

    def transform_condition_on_z(self, z, inputs, gridpoints=None, gridpoints_idx=None, feature_vec=None):
        """

        :param z: an integer
        :param inputs: (T_pre, D)
        :param gridpoints_idx:
        :param feature_vec:
        :return:
        """

        _, D = inputs.shape
        assert D == self.D, "input should have last dimension = {}".format(self.D)

        if gridpoints_idx is None:
            gridpoints_idx_a = self.get_gridpoints_idx_for_single(inputs[-1, 0:2])  # (4,)
            gridpoints_idx_b = self.get_gridpoints_idx_for_single(inputs[-1, 2:4])  # (4,)
        else:
            assert isinstance(gridpoints_idx, tuple)
            gridpoints_idx_a, gridpoints_idx_b = gridpoints_idx
            #print("Using grid points idx memory!")
        assert gridpoints_idx_a.shape == (self.GP, 4)
        assert gridpoints_idx_b.shape == (self.GP, 4)

        if gridpoints is None:
            gridpoints_a = self.get_gridpoints_for_single(gridpoints_idx_a)
            gridpoints_b = self.get_gridpoints_for_single(gridpoints_idx_b)
        else:
            assert isinstance(gridpoints, tuple)
            gridpoints_a, gridpoints_b = gridpoints
            #print("Using grid points memory!")
        assert gridpoints_a.shape == (self.d, 2)
        assert gridpoints_b.shape == (self.d, 2)

        weights_a = self.get_weights_for_single(inputs[-1, 0:2], 0, gridpoints_a, gridpoints_idx_a, z)  # (1, Df)
        weights_b = self.get_weights_for_single(inputs[-1, 2:4], 1, gridpoints_b, gridpoints_idx_b, z)  # (1, Df)
        assert weights_a.shape == (1, self.Df)
        assert weights_b.shape == (1, self.Df)

        if feature_vec is None:
            feature_vec_a = self.feature_vec_func(inputs[-1:, 0:2])
            feature_vec_b = self.feature_vec_func(inputs[-1:, 2:4])
        else:
            feature_vec_a, feature_vec_b = feature_vec
            #print("Using feature vec memory!")

        assert feature_vec_a.shape == (1, self.Df, self.d)
        assert feature_vec_b.shape == (1, self.Df, self.d)

        # (1, Df), (1, Df, d) --> (1, 1, d)
        out_a = torch.matmul(torch.sigmoid(weights_a), feature_vec_a)  # (1, 1, 2)
        out_b = torch.matmul(torch.sigmoid(weights_b), feature_vec_b)  # (1, 1, 2)
        assert out_a.shape == (1, 1, self.d)
        assert out_b.shape == (1, 1, self.d)

        out = torch.cat((out_a, out_b), dim=-1)  # (1, 1, 4)
        out = torch.squeeze(out)  # (4,)

        out = inputs[-1] + self.acc_factor * out
        assert out.shape == (self.D, )
        return out

    def get_gridpoints_idx_for_single(self, point):
        """

        :param point: (2,)
        :return: idx (GP, 4)
        """
        assert point.shape == (2, )
        find = False

        idx = torch.zeros((self.GP, 4), dtype=torch.float64)

        l_y = len(self.y_grids)
        for i in range(len(self.x_grids)-1):
            for j in range(len(self.y_grids)-1):
                cond_x = self.x_grids[i] <= point[0] <= self.x_grids[i+1]
                cond_y = self.y_grids[j] <= point[1] <= self.y_grids[j+1]
                if cond_x & cond_y:
                    find = True
                    idx[i*l_y+j, 0] = 1  # Q11
                    idx[i*l_y+j+1, 1] = 1  # Q12
                    idx[(i+1)*l_y+j, 2] = 1  # Q21
                    idx[(i+1)*l_y+j+1, 3] = 1  # Q22
                    break
            if find:
                break
        if not find:
            raise ValueError("value {} out of the grid world.".format(point.numpy()))
        return idx

    def get_gridpoints_idx_for_batch(self, points):
        idx = list(map(self.get_gridpoints_idx_for_single, points))
        idx = torch.stack(idx, dim=0)
        return idx

    def get_gridpoints_for_single(self, idx):
        """

        :param idx: (GP, 4)
        :return: gridpoints: (d, 2)
        """
        # (d, GP) * (GP, 2) --> (d, 2)
        out = torch.matmul(self.gridpoints, idx[:, [0, -1]])
        return out

    def get_gridpoints_for_batch(self, idx):
        """

        :param idx: (T, GP, 4)
        :return: gridpoints: (T, d, 2)
        """
        # (d, GP) * (T, GP, 4) --> (T, d, 2)
        out = torch.matmul(self.gridpoints, idx[:, :, [0, -1]])
        return out

    def get_weights_for_single(self, point, animal_idx, grid_points, grid_points_idx, z):
        """

        :param point: (2, )
        :param animal_idx: 0 or 1
        :param grid_points: (d, 2)
        :param grid_points_idx: (GP, 4)
        :param z: scalar
        :return: (1, Df)
        """
        assert point.shape == (self.d, )
        assert grid_points.shape == (self.d, 2)

        # (Df,GP) * (GP, 4) -->  (Df, 4)
        grid_points_weights = torch.matmul(torch.transpose(self.Ws[z, animal_idx], 0, 1), grid_points_idx)

        weight = two_d_interpolation(point, grid_points[:,0], grid_points[:,1],
                                     grid_points_weights[:,0][None,], grid_points_weights[:,1][None, ],
                                     grid_points_weights[:,2][None, ], grid_points_weights[:,3][None, ])
        assert weight.shape == (1, self.Df)
        return weight

    def get_weights_for_batch(self, points, animal_idx, grid_points, grid_points_idx):
        """

        :param points: (T, 2)
        :param animal_idx: 0 or 1
        :param grid_points: nearby grid points: (T, d, 2)
        :param grid_points_idx: (T, GP, 4)
        :return: (T, Df)
        """
        T, d = points.shape
        assert d == self.d

        # (1, K, Df, GP)* (T, 1, GP, 4)  --> (T, K, Df, 4)
        grid_point_weights = torch.matmul(torch.transpose(self.Ws[:, animal_idx], 1, 2)[None, ], grid_points_idx[:, None])

        # (T, K, Df)
        weights = two_d_interpolation(points, grid_points[..., 0], grid_points[..., 1],
                                      grid_point_weights[..., 0], grid_point_weights[..., 1],
                                      grid_point_weights[..., 2], grid_point_weights[..., 3])

        assert weights.shape == (T, self.K, self.Df)
        return weights


def one_d_interpolation(x, x1, x2, f_x1, f_x2):
    """

    :param x: (T, 1)  or (1, )
    :param x1: (T, 1) or (1, )
    :param x2: (T, 1) or (1, )
    :param f_x1: (T, K, Df) or (K, Df)
    :param f_x2: (T, K, Df) or (K, Df)
    :return: (T, K, Df) or (K, Df)
    """

    factor_1 = (x2 - x) / (x2 - x1)  # (T, 1) or (1, )
    factor_2 = (x - x1) / (x2 - x1)  # (T, 1) or (1, )

    out =  factor_1[..., None] * f_x1 + factor_2[..., None] * f_x2
    assert out.shape == f_x1.shape
    return out


def two_d_interpolation(points, Q11, Q22, f_Q11, f_Q12, f_Q21, f_Q22):
    """

    :param points: (T, 2) or (2, )
    :param Q11: (T, 2) or (2, )
    :param Q22: (T, 2) or (2, )
    :param f_Q11: (T, K, Df) or (K, Df)
    :param f_Q12: (T, K, Df) or (K, Df)
    :param f_Q21: (T, K, Df) or (K, Df)
    :param f_Q22: (T, K, Df) or (K, Df)
    :return: (T, K, Df)
    """

    x, y = points[..., 0:1], points[..., 1:2]  # each is (T, 1) or (1, )
    x1, y1 = Q11[..., 0:1], Q11[..., 1:2]  # each is (T, 1) or (1, )
    x2, y2 = Q22[..., 0:1], Q22[..., 1:2]  # each is (T, 1) or (1, )

    f_Qxy1 = one_d_interpolation(x, x1, x2, f_Q11, f_Q21)  # (T, K, Df) or (K, Df)
    f_Qxy2 = one_d_interpolation(x, x1, x2, f_Q12, f_Q22)  # (T, K, Df) or (K, Df)
    out = one_d_interpolation(y, y1, y2, f_Qxy1, f_Qxy2)  # (T, K, Df) or(K, Df)
    assert out.shape == f_Q11.shape
    return out

