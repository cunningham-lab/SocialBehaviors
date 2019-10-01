import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import set_param, check_and_convert_to_tensor, random_rotation


class DynamicLocationTransformation(BaseTransformation):
    """
    Learnable parameters: weights of each grid vertex
    weights of any random point = 2D linear interpolation of the nearby four grid points
    """

    def __init__(self, K, D):
        super(DynamicLocationTransformation, self).__init__(K, D)

        self.d = int(self.D / 2)

        # dynamic parameters
        self.As = 0.95 * torch.ones((self.K, self.D, self.d), dtype=torch.float64, requires_grad=True)
        self.bs = 0.1 * torch.ones((self.K, self.D), dtype=torch.float64, requires_grad=True)
        self.log_sigmas_dyn = torch.tensor(np.log(np.ones((K, D))), requires_grad=True)

        # location parameters
        self.mus_loc = torch.eye(self.K, self.D, dtype=torch.float64, requires_grad=True)
        self.log_sigmas_loc = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return self.As, self.bs, self.log_sigmas_dyn, self.mus_loc, self.log_sigmas_loc

    @params.setter
    def params(self, values):
        self.As = set_param(self.As, values[0])
        self.bs = set_param(self.bs, values[1])
        self.log_sigmas_dyn = set_param(self.log_sigmas_dyn, values[2])
        self.mus_loc = set_param(self.mus_loc, values[3])
        self.log_sigmas_loc = set_param(self.log_sigmas_loc, values[4])

    def permute(self, perm):
        self.As = self.As[perm]
        self.bs = self.bs[perm]
        self.log_sigmas_dyn = self.log_sigmas_dyn[perm]
        self.mus_loc = self.mus_loc[perm]
        self.log_sigmas_loc = self.log_sigmas_loc[perm]

    @property
    def log_sigma(self):
        # (T, K, D)
        return self.log_sigmas_dyn + self.log_sigmas_loc \
               - 1 / 2 * torch.log(torch.exp(self.log_sigmas_dyn)**2 + torch.exp(self.log_sigmas_loc)**2)

    @property
    def sigma_loc_square(self):
        return torch.exp(self.log_sigmas_loc)**2

    @property
    def sigma_dyn_square(self):
        return torch.exp(self.log_sigmas_dyn)**2

    def transform(self, inputs, **kwargs):
        """

        :param inputs: (T, D)
        :return:
        """

        T, D = inputs.shape
        assert D == self.D, "input should have last dimension = {}".format(self.D)

        # Ax + b
        # (K, 2, 2) * (T, 1, 2,1) --> (T, K, 2, 1)
        out_a = torch.matmul(self.As[:, 0:2], inputs[:, None, 0:2, None])
        out_b = torch.matmul(self.As[:, 2:4], inputs[:, None, 2:4, None])
        mu_dyn = torch.cat((out_a, out_b), dim=-2)  # (T, K, 4, 1)
        mu_dyn = torch.squeeze(mu_dyn, dim=-1)  # (T, K, D)
        assert mu_dyn.shape == (T, self.K, self.D)

        # (K, D) * (T, K, D) --> (T, K, D)
        # (K, D) * (K, D) --> (K, D)
        # (T, K, D) + (K, D) --> (T, K, D)
        mu_combined = self.sigma_loc_square * mu_dyn + self.sigma_dyn_square * self.mus_loc
        mu_combined = mu_combined / (self.sigma_loc_square + self.sigma_dyn_square)
        assert mu_combined.shape == (T, self.K, self.D)

        return mu_combined

    def transform_condition_on_z(self, z, inputs):
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


