import torch
import numpy as np
import numpy.random as npr

from ssm_ptc.transitions.base_transition import BaseTransition
from ssm_ptc.utils import set_param, get_np


class GridTransition(BaseTransition):
    """
    each grid has its own transition matrices
    """
    def __init__(self, K, D, M, x_grids, y_grids, Pis=None, device=torch.device('cpu')):
        super(GridTransition, self).__init__(K=K, D=D, M=M, device=device)

        self.x_grids = x_grids  # [x0, ..., xn]
        self.y_grids = y_grids  # [y0, ..., ym]

        self.n_grids = (len(self.x_grids) - 1) * (len(self.y_grids) - 1)
        if Pis is None:
            Pis = [2 * np.eye(self.K) + .05 * npr.rand(self.K, self.K)
                   for _ in range(self.n_grids) for _ in range(self.n_grids)]

        self.Pis = torch.tensor(Pis, dtype=torch.float64, requires_grad=True, device=self.device)
        assert self.Pis.shape == (self.n_grids, self.n_grids, self.K, self.K), \
            "shape should be {} instead of {}".format((self.n_grids, self.K, self.K), self.Pis.shape)

    @property
    def params(self):
        return self.Pis,

    @params.setter
    def params(self, values):
        self.Pis = set_param(self.Pis, values[0])

    def permute(self, perm):
        Pis = self.Pis.detach().numpy()  # (n_grids, n_grids, K, K)
        Pis = np.reshape(Pis, (-1, self.K, self.K))
        Pis = [Pi[np.idx(perm, perm)] for Pi in Pis]
        Pis = np.reshape(Pis, (self.n_grids, self.n_grids, self.K, self.K))
        self.Pi = torch.tensor(Pis, dtype=torch.float64, requires_grad=True, device=self.device)

    def transition_matrix(self, data, input, log=False, **kwargs):
        """

        :param data: [T, D]
        :param input:
        :param log:
        :return: (T, K, K)
        """
        T, _ = data.shape
        # for each data point, find out which transition to use
        joint_grid_idx = kwargs.get("joint_grid_idx", None)
        if joint_grid_idx is None:
            joint_grid_idx = self.get_joint_grid_idx(data)

        Pis = self.Pis[joint_grid_idx]
        Pis = torch.nn.LogSoftmax(dim=-1)(Pis)
        assert Pis.shape == (T, self.K, self.K)
        return Pis

    def get_joint_grid_idx(self, data, **kwargs):
        """
        find out the joint-grid that each data point belongs to
        :param data: (T, D)
        :return: joint_grid_idx, [idx_a, idx_b], each is a list of length T
        """

        idx_a = kwargs.get("idx_a", None)
        idx_b = kwargs.get("idx_b", None)

        if idx_a is None:
            idx_a = list(map(self.get_grid_idx_for_single, data[:,0:2]))

        if idx_b is None:
            idx_b = list(map(self.get_grid_idx_for_single, data[:,2:4]))

        return [idx_a, idx_b]

    def get_grid_idx_for_single(self, point):
        """

        :param point: (2,)
        :return: grid idx: a scalar
        """
        assert point.shape == (2,)
        find = False

        grid_idx = 0
        for i in range(len(self.x_grids) - 1):
            for j in range(len(self.y_grids) - 1):
                cond_x = self.x_grids[i] <= point[0] <= self.x_grids[i + 1]
                cond_y = self.y_grids[j] <= point[1] <= self.y_grids[j + 1]
                if cond_x & cond_y:
                    find = True
                    break
                grid_idx += 1
            if find:
                break
        if not find:
            raise ValueError("value {} out of the grid world.".format(get_np(point)))
        return grid_idx