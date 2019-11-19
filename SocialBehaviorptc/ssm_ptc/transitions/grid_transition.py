import torch
import numpy as np
import numpy.random as npr

from ssm_ptc.transitions.base_transition import BaseTransition
from ssm_ptc.utils import set_param, get_np

# TODO: test single first
class GridTransition(BaseTransition):
    """
    each grid has its own transition matrices
    """
    def __init__(self, K, D, M, x_grids, y_grids, Pi=None, device=torch.device('cpu')):
        super(GridTransition, self).__init__(K=K, D=D, M=M, device=device)
        assert D == 2 or D == 4, D

        self.x_grids = x_grids  # [x0, ..., xn]
        self.y_grids = y_grids  # [y0, ..., ym]

        self.n_grids = (len(self.x_grids) - 1) * (len(self.y_grids) - 1)
        Pis = Pi
        if self.D == 4:
            if Pis is None:
                Pis = [2 * np.eye(self.K) + .05 * npr.rand(self.K, self.K)
                       for _ in range(self.n_grids) for _ in range(self.n_grids)]
            self.Pis = torch.tensor(Pis, dtype=torch.float64, requires_grad=True, device=self.device)
            assert self.Pis.shape == (self.n_grids, self.n_grids, self.K, self.K), \
                "shape should be {} instead of {}".format((self.n_grids, self.n_grids, self.K, self.K), self.Pis.shape)
        else:
            if Pis is None:
                Pis = [2 * np.eye(self.K) + .05 * npr.rand(self.K, self.K)
                       for _ in range(self.n_grids)]
            self.Pis = torch.tensor(Pis, dtype=torch.float64, requires_grad=True, device=self.device)
            assert self.Pis.shape == (self.n_grids, self.K, self.K), \
                "shape should be {} instead of {}".format((self.n_grids, self.K, self.K), self.Pis.shape)

    @property
    def params(self):
        return self.Pis,

    @params.setter
    def params(self, values):
        self.Pis = set_param(self.Pis, values[0])

    @property
    def grid_transition_matrix(self):
        return torch.nn.Softmax(dim=-1)(self.Pis)

    def permute(self, perm):
        Pis = self.Pis.detach().numpy()  # (n_grids, n_grids, K, K)
        if self.D == 4:
            Pis = np.reshape(Pis, (-1, self.K, self.K))
            Pis = [Pi[np.idx(perm, perm)] for Pi in Pis]
            Pis = np.reshape(Pis, (self.n_grids, self.n_grids, self.K, self.K))
        else:
            Pis = [Pi[np.idx(perm, perm)] for Pi in Pis]
        self.Pi = torch.tensor(Pis, dtype=torch.float64, requires_grad=True, device=self.device)

    def transition_matrix(self, data, input, log=False, **kwargs):
        """

        :param data: [T, D]
        :param input:
        :param log:
        :return: (T, K, K)
        """
        T, D = data.shape
        assert D == 4 or D == 2, D
        # for each data point, find out which transition to use
        joint_grid_idx = kwargs.get("joint_grid_idx", None)
        if joint_grid_idx is None:
            print("not using transition memory!")
            if self.D == 4:
                joint_grid_idx = self.get_joint_grid_idx(data)
            else:
                joint_grid_idx = self.get_grid_idx(data)

        Pi = self.Pis[joint_grid_idx]
        if log:
            Pi = torch.nn.LogSoftmax(dim=-1)(Pi)
        else:
            Pi = torch.nn.Softmax(dim=-1)(Pi)
        assert Pi.shape == (T, self.K, self.K)
        return Pi

    def log_transition_matrix(self, data, input, **kwargs):
        return self.transition_matrix(data, input, log=True, **kwargs)

    def get_joint_grid_idx(self, data, **kwargs):
        """
        find out the joint-grid that each data point belongs to
        :param data: (T, 4)
        :return: joint_grid_idx, [idx_a, idx_b], each is a list of length T
        """

        idx_a = kwargs.get("idx_a", None)
        idx_b = kwargs.get("idx_b", None)

        if idx_a is None:
            idx_a = self.get_grid_idx(data[:,0:2])

        if idx_b is None:
            idx_b = self.get_grid_idx(data[:,2:4])

        return [idx_a, idx_b]

    def get_grid_idx(self, data):
        """

        :param data: (T, 2)
        :return: a list
        """
        idx = list(map(self.get_grid_idx_for_single, data))
        return idx

    def get_grid_idx_for_single(self, point):
        """

        :param point: (2,)
        :return: grid idx: a scalar
        """
        assert point.shape == (2,), point.shape
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
