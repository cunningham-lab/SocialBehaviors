import torch
import numpy as np

from ssm_ptc.utils import check_and_convert_to_tensor, set_param, get_np

from project_ssms.coupled_transformations.base_weighted_direction_transformation \
    import BaseWeightedDirectionTransformation

# label the gpts from down to up, from left right
# the order of four gpts in one grid
# see get_gpt_idx


class GPGridTransformation(BaseWeightedDirectionTransformation):
    """
    Learnable parameters: weights of each grid vertex
    weights of any random point = a weighted combination of weights of all the grid vertices
    where the weights = softmax
    """
    def __init__(self, K, D, x_grids, y_grids, Df, feature_vec_func, acc_factor=2,
                rs=None, rs_factor=None, train_rs=False, lags=1,
                 device=torch.device('cpu'), version=1):
        assert lags == 1, "lags should be 1 for weigthedgird transformation"
        super(GPGridTransformation, self).__init__(K, D, Df, feature_vec_func, acc_factor, lags=lags)

        self.version = version

        self.device = device
        self.x_grids = check_and_convert_to_tensor(x_grids, dtype=torch.float64, device=self.device)  # [x_0, x_1, ..., x_m]
        self.y_grids = check_and_convert_to_tensor(y_grids, dtype=torch.float64, device=self.device)  # a list [y_0, y_1, ..., y_n]

        # shape: (d, n_gps)
        self.gridpoints = torch.tensor([(x_grid, y_grid) for x_grid in self.x_grids for y_grid in self.y_grids],
                                       device=self.device)  # (n_gps, 2)

        # number of basis grid points
        self.n_gps = self.gridpoints.shape[0]
        # number of grids
        self.n_grids = (len(self.x_grids)-1) * (len(self.y_grids)-1)

        self.Ws = torch.rand(self.K, 2, self.n_gps, self.Df, dtype=torch.float64, device=self.device,
                             requires_grad=True)

        self.gpts_pairwise_xydist_sq = self.get_gpts_pariwise_xydist_sq()
        assert self.gpts_pairwise_xydist_sq.shape == (self.n_grids, 4, 4, 2), self.gpts_pairwise_xydist_sq.shape

        # define n_gps parameters, suppose parameters work for all Ks
        if rs_factor is None:
            x_grids = get_np(x_grids)
            y_grids = get_np(y_grids)
            x_gap = (x_grids[-1] - x_grids[0])/(len(x_grids)-1)
            y_gap = (y_grids[-1] - y_grids[0])/(len(y_grids)-1)
            rs_factor = np.array([x_gap, y_gap])
        self.rs_factor = check_and_convert_to_tensor(rs_factor, device=self.device)
        if rs is None:
            rs = np.array([10, 10])
        self.rs = torch.tensor(rs, dtype=torch.float64, device=self.device, requires_grad=train_rs)
        # real_rs = rs_factor * torch.sigmoid(rs)

    def get_gpts_pariwise_xydist_sq(self):
        gpts_pairwise_xydist_sq = []

        l_y = len(self.y_grids)
        for i in range(len(self.x_grids) - 1):
            for j in range(len(self.y_grids) - 1):
                Q11= self.gridpoints[i * l_y + j]  # Q11
                Q12 = self.gridpoints[i * l_y + j + 1]  # Q12
                Q21 = self.gridpoints[(i + 1) * l_y + j]  # Q21
                Q22 = self.gridpoints[(i + 1) * l_y + j + 1]  # Q22
                Qs = torch.stack([Q11, Q12, Q21, Q22], dim=0)
                xydist_sq = pairwise_xydist_sq(Qs, Qs)
                assert xydist_sq.shape == (4, 4, 2)
                gpts_pairwise_xydist_sq.append(xydist_sq)
        gpts_pairwise_xydist_sq = torch.stack(gpts_pairwise_xydist_sq, dim=0)
        return gpts_pairwise_xydist_sq

    @property
    def params(self):
        return self.Ws, self.rs

    @params.setter
    def params(self, values):
        self.Ws = set_param(self.Ws, values[0])
        self.rs = set_param(self.rs, values[1])

    def permute(self, perm):
        self.Ws = self.Ws[perm]
        self.rs = self.rs[perm]

    def get_weights(self, inputs, **kwargs):
        weights_a = self.get_weights_for_single_animal(inputs[:, 0:2], 0, **kwargs)
        weights_b = self.get_weights_for_single_animal(inputs[:, 2:4], 1, **kwargs)
        return weights_a, weights_b

    def get_weights_for_single_animal(self, inputs, animal_idx, **kwargs):
        """
        w(x) = K_{xG} (K_G + \sigma^2 I)^{-1} w(G) where G=the nearby (g1,g2,g3,g4)^T

        If vs and rs are fixed, then gp coeff should be fixed, kwargs should supply coeff. Else, gp coeff is not fixed,
        kwargs should supply dist
        :param inputs: (T, 2)
        :param animal_idx: 0 or 1
        :param kwargs:
        :return: weights of inputs, shape (T, K, Df)
        """
        T, d = inputs.shape
        assert d == self.d, "inputs should have shape {}, instead of {}".format((T, self.d), (T, d))

        gpt_idx_key, grid_idx_key = ("gpt_idx_a", "grid_idx_a") if animal_idx == 0 else ("gpt_idx_b", "grid_idx_b")

        gpt_idx_s = kwargs.get(gpt_idx_key, None)
        grid_idx_s = kwargs.get(grid_idx_key, None)
        if gpt_idx_s is None or grid_idx_s is None:
            #print("not providing gpt_idx and grid_idx memories")
            gpt_idx_s, grid_idx_s = self.get_gpt_idx_and_grid_idx_for_batch(inputs)
        assert gpt_idx_s.shape == (T, 4), gpt_idx_s.shape
        assert grid_idx_s.shape == (T, ), grid_idx_s.shape

        coeff_key = "coeff_a" if animal_idx == 0 else "coeff_b"
        coeff = kwargs.get(coeff_key, None)
        if coeff is None:
            #print("not providing coefficients memories")
            coeff = self.get_gp_coefficients(inputs, animal_idx, gpt_idx_s, grid_idx_s, **kwargs)
        assert coeff.shape == (T, 1, 4)

        # Ws shape = (K, n_gps, Df)
        weights_of_four_grids = self.Ws[:, animal_idx, gpt_idx_s]
        assert weights_of_four_grids.shape == (self.K, T, 4, self.Df)
        weights_of_four_grids = torch.transpose(weights_of_four_grids, 0, 1)
        assert weights_of_four_grids.shape == (T, self.K, 4, self.Df)

        if self.version == 1:
            # (T, 1, 1, 4) * (T, K, 4, Df) -> (T, K, 1, Df)
            weigths_of_inputs = torch.matmul(coeff[:,None], weights_of_four_grids)
            weigths_of_inputs = self.acc_factor * torch.sigmoid(weigths_of_inputs)
        elif self.version == 2:
            weights_of_four_grids = self.acc_factor * torch.sigmoid(weights_of_four_grids)
            # (T, 1, 1, 4) * (T, K, 4, Df) -> (T, K, 1, Df)
            weigths_of_inputs = torch.matmul(coeff[:, None], weights_of_four_grids)
        else:
            raise ValueError("invalid version: {}".format(self.version))

        assert weigths_of_inputs.shape == (T, self.K, 1, self.Df), \
            "weights_of_inputs should have shape {}, instead of {}".format((T, self.K, 1, self.Df),
                                                                           weigths_of_inputs.shape)
        weigths_of_inputs = torch.squeeze(weigths_of_inputs, dim=2)

        return weigths_of_inputs

    def get_weights_condition_on_z(self, inputs, z, **kwargs):
        weights_a = self.get_weights_condition_on_z_for_single_animal(inputs[:, 0:2], animal_idx=0, z=z, **kwargs)
        weights_b = self.get_weights_condition_on_z_for_single_animal(inputs[:, 2:4], animal_idx=1, z=z, **kwargs)
        return weights_a, weights_b

    def get_weights_condition_on_z_for_single_animal(self, inputs, animal_idx, z, **kwargs):
        """
        w(x) = K_{xG} (K_G + \sigma^2 I)^{-1} w(G) where G=the nearby (g1,g2,g3,g4)^T

        If vs and rs are fixed, then gp coeff should be fixed, kwargs should supply coeff. Else, gp coeff is not fixed,
        kwargs should supply dist
        :param inputs: (T+pre, 2)
        :param animal_idx: 0 or 1
        :param kwargs:
        :return: weights of inputs, shape (1, Df)
        """
        _, d = inputs.shape
        assert d == self.d, "inputs should have last dim of shape {}, instead of {}".format(self.d, d)

        gpt_idx_key, grid_idx_key = ("gpt_idx_a", "grid_idx_a") if animal_idx == 0 else ("gpt_idx_b", "grid_idx_b")

        gpt_idx_s = kwargs.get(gpt_idx_key, None)
        grid_idx_s = kwargs.get(grid_idx_key, None)
        if gpt_idx_s is None or grid_idx_s is None:
            #print("not providing gpt_idx and grid_idx memories")
            gpt_idx_s, grid_idx_s = self.get_gpt_idx_and_grid_idx_for_batch(inputs[-1:])
        assert gpt_idx_s.shape == (1, 4), gpt_idx_s.shape
        assert grid_idx_s.shape == (1, ), grid_idx_s.shape

        coeff_key = "coeff_a" if animal_idx == 0 else "coeff_b"
        coeff = kwargs.get(coeff_key, None)
        if coeff is None:
            #print("not providing coefficients memories")
            coeff = self.get_gp_coefficients(inputs[-1:], animal_idx, gpt_idx_s, grid_idx_s, **kwargs)
        assert coeff.shape == (1, 1, 4)

        # Ws shape = (n_gps, Df)
        weights_of_four_grids = self.Ws[z, animal_idx, gpt_idx_s[0]]
        assert weights_of_four_grids.shape == (4, self.Df)

        if self.version == 1:
            # (1, 4) * (4, Df) -> (1, 1, Df)
            weigths_of_inputs = torch.matmul(coeff[0], weights_of_four_grids)
            weigths_of_inputs = self.acc_factor * torch.sigmoid(weigths_of_inputs)
        elif self.version == 2:
            weights_of_four_grids = self.acc_factor * torch.sigmoid(weights_of_four_grids)
            # (1, 4) * (4, Df) -> (1, 1, Df)
            weigths_of_inputs = torch.matmul(coeff[0], weights_of_four_grids)
        else:
            raise ValueError("invalid version: {}".format(self.version))

        assert weigths_of_inputs.shape == (1, self.Df), \
            "weights_of_inputs should have shape {}, instead of {}".format((1, self.Df),
                                                                           weigths_of_inputs.shape)

        return weigths_of_inputs

    def get_gp_coefficients(self, points, animal_idx, gpt_idx_s, grid_idx_s, **kwargs):
        """

        :param points: (T, 2)
        :param animal_idx: 0 or 1
        :param gpt_idx_s: (T, 4)
        :param grid_idx_s: (T, )
        :param kwargs:
        :return: (T, K, 4)
        """
        T = len(points)
        dist_sq_key = "dist_sq_a" if animal_idx == 0 else "dist_sq_b"
        dist_sq_s = kwargs.get(dist_sq_key, None)
        if dist_sq_s is None:
            #print("not providing dist_sq memories")
            nearby_gpts = self.gridpoints[gpt_idx_s]
            assert nearby_gpts.shape == (T, 4, 2)
            dist_sq_s = (points[:,None] - nearby_gpts)**2  # (T, 4, 2)
        assert dist_sq_s.shape == (T, 4, 2), \
            "dist_sq_s should have shape {}, instead of {}".format((T, self.n_gps), dist_sq_s.shape)

        real_rs_sq = (self.rs_factor*torch.sigmoid(self.rs)) ** 2  # (2,)

        # coefficients of the grid weights: (T, K, n_gps)
        # (T, 4, 2) * (2,1) --> (T, 4, 1)
        K_xG = torch.exp(-1 / 2 * torch.matmul(dist_sq_s, 1/real_rs_sq[:,None]))  # (T, 4,1)
        assert K_xG.shape == (T, 4, 1), K_xG.shape
        K_xG = torch.squeeze(K_xG, dim=-1)

        # (T, 4, 4, 2) * (2,1) -> (T, 4, 4, 1)
        tmp = self.gpts_pairwise_xydist_sq[grid_idx_s]
        assert tmp.shape == (T, 4, 4, 2)

        # (n_grids, 4, 4, 2) * (2,1) --> (n_grids, 4, 4, 1)
        kernels = torch.exp(- 1 / 2 * torch.matmul(self.gpts_pairwise_xydist_sq, 1/real_rs_sq[:,None]))
        assert kernels.shape == (self.n_grids, 4, 4, 1)
        kernels = torch.squeeze(kernels, dim=-1)

        inverse_kernels = torch.inverse(kernels)
        assert inverse_kernels.shape == (self.n_grids, 4,4)

        batch_inverse_kernels = inverse_kernels[grid_idx_s]  # (T, 4, 4)
        assert batch_inverse_kernels.shape == (T, 4, 4), batch_inverse_kernels.shape

        coeff = torch.matmul(K_xG[:, None], batch_inverse_kernels)
        assert coeff.shape == (T, 1, 4)
        return coeff

    def get_gpt_idx_and_grid_for_single(self, point):
        """

        :param point: (2,)
        :return: gpt_idx a list of length 4
        """
        assert point.shape == (2,)
        find = False

        gpt_idx = []
        grid_idx = 0
        l_y = len(self.y_grids)
        for i in range(len(self.x_grids) - 1):
            for j in range(len(self.y_grids) - 1):
                cond_x = self.x_grids[i] <= point[0] <= self.x_grids[i + 1]
                cond_y = self.y_grids[j] <= point[1] <= self.y_grids[j + 1]
                if cond_x & cond_y:
                    find = True
                    gpt_idx.append(i * l_y + j)  # Q11
                    gpt_idx.append(i * l_y + j + 1)  #  (Q12)
                    gpt_idx.append((i + 1) * l_y + j)  # (Q21)
                    gpt_idx.append((i + 1) * l_y + j + 1)  # Q22
                    break
                grid_idx += 1
            if find:
                break
        if not find:
            raise ValueError("value {} out of the grid world.".format(get_np(point)))
        return gpt_idx, grid_idx

    def get_gpt_idx_and_grid_idx_for_batch(self, points):
        """

        :param points: (T, 2)
        :return: idx: (T, 4)
        """
        out = list(map(self.get_gpt_idx_and_grid_for_single, points))
        idx = [out_i[0] for out_i in out]
        grid_idx = [out_i[1] for out_i in out]
        idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        grid_idx = torch.tensor(grid_idx, dtype=torch.long, device=self.device)

        return idx, grid_idx

def pairwise_xydist_sq(points_a, points_b):
    """

    :param points_a: (n1, 2)
    :param points_b: (n2, 2)
    :return: x_dist_sq, y_dist_sq, each of shape (n1, n2)
    """
    n1, d = points_a.shape
    assert d == 2, "points_a should have shape {}, instead of {}".format((n1, 2), (n1, d))

    n2, d = points_b.shape
    assert d == 2, "points_b should have shape {}, instead of {}".format((n2, 2), (n2, d))

    x_dist = points_a[:, None, 0] - points_b[None, :, 0]
    assert x_dist.shape == (n1, n2)

    y_dist = points_a[:, None, 1] - points_b[None, :, 1]

    xy_dist_sq = torch.stack([x_dist**2, y_dist**2], dim=-1)

    assert xy_dist_sq.shape == (n1, n2, 2)
    return xy_dist_sq
