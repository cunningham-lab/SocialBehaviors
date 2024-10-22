import torch
import torch.nn as nn
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import set_param, check_and_convert_to_tensor, get_np


class SingleLinearGridTransformation(BaseTransformation):
    """
    Learnable parameters: weights of each grid vertex
    weights of any random point = 2D linear interpolation of the nearby four grid points
    """

    def __init__(self, K, D, x_grids, y_grids,
                 device=torch.device('cpu')):
        super(SingleLinearGridTransformation, self).__init__(K, D)

        assert D == 2, D
        self.device = device

        self.x_grids = nn.Parameter(check_and_convert_to_tensor(x_grids, dtype=torch.float64), requires_grad=False)  # [x_0, x_1, ..., x_m]
        self.y_grids = nn.Parameter(check_and_convert_to_tensor(y_grids, dtype=torch.float64), requires_grad=False)  # a list [y_0, y_1, ..., y_n]
        self.n_x = len(x_grids) - 1
        self.n_y = len(y_grids) - 1
        self.ly = len(y_grids)
        # shape: (n_gps, D)
        self.gridpoints = np.array([(x_grid, y_grid) for x_grid in self.x_grids for y_grid in self.y_grids])
        # number of basis grid points
        self.n_gps = self.gridpoints.shape[0]

        # dynamics at the grid points
        self.us = nn.Parameter(torch.rand((self.K, self.n_gps, self.D), dtype=torch.float64), requires_grad=True)

    def transform(self, inputs, **kwargs):
        """
        do a 2D interpolation of weights, and use that for with_noise
        :param inputs: (T, D)
        :param kwargs:
        :return: (T, K, D)
        """

        T, D = inputs.shape
        assert D == self.D, "input should have last dimension = {}".format(self.D)

        coeffs = kwargs.get("coeffs", None)
        gridpoints_idx = kwargs.get("gridpoints_idx", None)
        if gridpoints_idx is None:
            gridpoints_idx = self.get_gridpoints_idx_for_batch(inputs)
        assert gridpoints_idx.shape == (T, 4), "shoulb be {}, but got {}".format((T, 4), gridpoints_idx.shape)
        if coeffs is None:
            gridpoints = self.gridpoints[gridpoints_idx]
            assert gridpoints.shape == (T, 4, 2)  # Q11, Q21, Q12, Q22
            coeffs = self.get_lp_coefficients(inputs, gridpoints[:,0], gridpoints[:,3], device=self.device)
        assert coeffs.shape == (T, 4)

        selected_us = self.us[:, gridpoints_idx]
        assert selected_us.shape == (self.K, T, 4, self.D), \
            "should be {}, but got {}".format((self.K, T, 4, self.D), selected_us.shape)
        selected_us = torch.transpose(selected_us, 0, 1)

        # (T, 1, 1, 4) * (T, K, 4, D) -> (T, K, 1, D)
        out = torch.matmul(coeffs[:, None, None], selected_us)
        assert out.shape == (T, self.K, 1, self.D), \
            "the correct shape is {}, but got {}.".format((T, self.K, 1, self.D), out.shape)
        out = torch.squeeze(out, dim=2)
        out = out + inputs[:, None]
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

        coeffs = kwargs.get("coeffs", None)
        gridpoints_idx = kwargs.get("gridpoints_idx", None)
        if gridpoints_idx is None:
            gridpoints_idx = self.get_gridpoints_idx_for_single_point(inputs[-1])
        assert len(gridpoints_idx) == 4, len(gridpoints_idx)
        if coeffs is None:
            gridpoints = self.gridpoints[gridpoints_idx]
            assert gridpoints.shape == (4, 2)  # Q11, Q21, Q12, Q22
            coeffs = self.get_lp_coefficients(inputs[-1:], gridpoints[0:1], gridpoints[3:4], device=self.device)
        assert coeffs.shape == (1, 4)

        # (T, K, D)
        selected_us = self.us[z][gridpoints_idx]
        assert selected_us.shape == (4, self.D), \
            "should be {}, but got {}".format((1, 4, self.D), selected_us.shape)

        # (1, 4) * (4, D) -> (1, D)
        out = torch.matmul(coeffs, selected_us)
        out = torch.squeeze(out, dim=0)
        out = out + inputs[-1]
        assert out.shape == (self.D, )
        return out

    def get_gridpoints_idx_for_single_point(self, point):
        """
        Q12 -- Q22
        |
        Q11 -- Q21
        :param point: (2,)
        :return: idx (4, ) idx for Q11, Q12, Q21, Q22)
        """
        assert point.shape == (2, )
        find = False

        for i in range(self.n_x):
            for j in range(self.n_y):
                cond_x = self.x_grids[i] <= point[0] <= self.x_grids[i + 1]
                cond_y = self.y_grids[j] <= point[1] <= self.y_grids[j + 1]
                if cond_x & cond_y:
                    find = True
                    idx = [i * self.ly + j, i * self.ly + j + 1,
                           (i + 1) * self.ly + j, (i + 1) * self.ly + j + 1]
                    break
            if find:
                break
        if not find:
            raise ValueError("value {} out of the grid world.".format(get_np(point)))
        return idx

    def get_gridpoints_idx_for_batch(self, points):
        idx = list(map(self.get_gridpoints_idx_for_single_point, points))
        return np.array(idx)

    @staticmethod
    def get_lp_coefficients(points, Q11s, Q22s, device=torch.device('cpu')):
        """
        Compute the coefficients of (Q11, Q12, Q21, Q22) for each point.
        :param points: (T, 2)
        :param Q11s: (T, 2)
        :param Q22s: (T, 2)
        :return:
        """
        Q11s = check_and_convert_to_tensor(Q11s, device=device)
        Q22s = check_and_convert_to_tensor(Q22s, device=device)
        T = points.shape[0]
        x2_minus_x1 = Q22s[:,0] - Q11s[:,0]
        y2_minus_y1 = Q22s[:,1] - Q11s[:,1]
        x2_minus_x = Q22s[:,0] - points[:,0]
        y2_minus_y = Q22s[:,1] - points[:,1]
        x_minus_x1 = points[:,0] - Q11s[:,0]
        y_minus_y1 = points[:,1] - Q11s[:,1]
        c_Q11 = x2_minus_x * y2_minus_y  # (T, )
        c_Q12 = x2_minus_x * y_minus_y1
        c_Q21 = x_minus_x1 * y2_minus_y
        c_Q22 = x_minus_x1 * y_minus_y1
        coeffs = torch.stack((c_Q11, c_Q12, c_Q21, c_Q22), dim=-1)  # (T, 4)
        coeffs = coeffs / ((x2_minus_x1 * y2_minus_y1)[:, None])
        assert coeffs.shape == (T, 4)
        return coeffs
