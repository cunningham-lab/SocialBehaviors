import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import set_param, check_and_convert_to_tensor, random_rotation


class SingleDynamicLocationTransformation(BaseTransformation):
    """
    Learnable parameters: weights of each grid vertex
    weights of any random point = 2D linear interpolation of the nearby four grid points
    """

    def __init__(self, K, D, bounds):
        assert D == 2
        super(SingleDynamicLocationTransformation, self).__init__(K, D)


        # dynamic parameters
        self.As = torch.tensor(0.001 * np.ones((self.K, self.D, self.D)), dtype=torch.float64, requires_grad=True)
        self.bs = torch.tensor(0.001 * np.ones((self.K, self.D)), dtype=torch.float64, requires_grad=True)
        self.log_sigmas_dyn = torch.tensor(np.log(np.ones((K, D))), requires_grad=True)

        # location parameters
        # random init between bounds
        assert bounds.shape == (D, 2), "bounds shape should be ({}, 2), instead of {}".format(D, bounds.shape)
        mus_val = np.random.rand(K, D)
        for d in range(D):
            mus_val[:, d] = (bounds[d, 1] - bounds[d, 0]) * mus_val[:, d] + bounds[d, 0]
        self.mus_loc = torch.tensor(mus_val, dtype=torch.float64, requires_grad=True)
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
    def log_sigmas(self):
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
        # (K, D, D) * (T, 1, D,1) --> (T, K, D, 1)
        out_a = torch.matmul(self.As, inputs[:, None, :, None])
        out_a = torch.squeeze(out_a, dim=-1)  # (T, K, D)
        mu_dyn = out_a + self.bs
        assert mu_dyn.shape == (T, self.K, self.D)

        # (K, D) * (T, K, D) --> (T, K, D)
        # (K, D) * (K, D) --> (K, D)
        # (T, K, D) + (K, D) --> (T, K, D)
        mu_combined = self.sigma_loc_square * mu_dyn + self.sigma_dyn_square * self.mus_loc
        mu_combined = mu_combined / (self.sigma_loc_square + self.sigma_dyn_square)
        assert mu_combined.shape == (T, self.K, self.D)

        return mu_combined

    def transform_condition_on_z(self, z, inputs, **kwargs):
        """

        :param z: an integer
        :param inputs: (T_pre, D)
        :param gridpoints_idx:
        :param feature_vec:
        :return:
        """

        _, D = inputs.shape
        assert D == self.D, "input should have last dimension = {}".format(self.D)

        # Ax + b
        # (D, D) * (D, 1) --> (D,1)
        out_a = torch.matmul(self.As[z], inputs[-1][:, None])

        assert out_a.shape == (self.D, 1)
        out = torch.squeeze(out_a, dim=-1)
        mu_dyn = out + self.bs[z]  # (D, )

        mu_combined = self.sigma_loc_square[z] * mu_dyn + self.sigma_dyn_square[z] * self.mus_loc[z]
        mu_combined = mu_combined / (self.sigma_loc_square[z] + self.sigma_dyn_square[z])

        assert mu_combined.shape == (self.D, )
        return mu_combined