import torch
from torch.distributions import Normal, MultivariateNormal
import numpy as np

from project_ssms.gp_observation import kernel_distsq
from project_ssms.utils import clip
from ssm_ptc.utils import check_and_convert_to_tensor, set_param, get_np
from ssm_ptc.observations.base_observation import BaseObservation


class SpatialTemporalGPSingle(BaseObservation):
    """
    Learnable parameters: weights of each grid vertex
    weights of any random point = a weighted combination of weights of all the grid vertices
    where the weights = softmax
    """
    def __init__(self, K, D, x_grids, y_grids, bounds, mus_init=None, log_sigmas_init=None, log_sigmas=None,
                rs=None, train_rs=False, train_vs=False, train_sigmas=False, mode='multiplicative',
                 device=torch.device('cpu')):
        assert D == 2
        super(SpatialTemporalGPSingle, self).__init__(K, D)

        self.device = device

        self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64, device=self.device)
        assert self.bounds.shape == (self.D, 2), self.bounds.shape

        # specify distribution parameters
        if mus_init is None:
            self.mus_init = torch.zeros(self.K, self.D, dtype=torch.float64, device=self.device)
        else:
            self.mus_init = check_and_convert_to_tensor(mus_init, dtype=torch.float64, device=self.device)
        # consider diagonal covariance
        if log_sigmas_init is None:
            self.log_sigmas_init = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, device=self.device)
        else:
            self.log_sigmas_init = check_and_convert_to_tensor(log_sigmas_init, dtype=torch.float64, device=self.device)
        assert self.log_sigmas_init.shape == (self.K, self.D)
        if log_sigmas is None:
            self.log_sigmas = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, device=self.device,
                                           requires_grad=train_sigmas)
        else:
            self.log_sigmas = check_and_convert_to_tensor(log_sigmas, dtype=torch.float64, device=self.device,
                                                          requires_grad=train_sigmas)
        assert self.log_sigmas.shape == (self.K, self.D)

        # specify gp dynamics parameters
        self.x_grids = check_and_convert_to_tensor(x_grids, dtype=torch.float64, device=self.device)  # [x_0, x_1, ..., x_m]
        self.y_grids = check_and_convert_to_tensor(y_grids, dtype=torch.float64, device=self.device)  # a list [y_0, y_1, ..., y_n]

        self.inducing_points = torch.tensor([(x_grid, y_grid) for x_grid in self.x_grids for y_grid in self.y_grids],
                                            device=self.device)  # (n_gps, 2)
        self.n_gps = self.inducing_points.shape[0]

        self.us = torch.rand(self.K, self.n_gps, self.D, dtype=torch.float64, device=self.device, requires_grad=True)

        # define n_gps parameters, suppose parameters work for all Ks
        if rs is None:
            x_grids = get_np(x_grids)
            y_grids = get_np(y_grids)
            x_gap = (x_grids[-1] - x_grids[0])/(len(x_grids)-1)
            y_gap = (y_grids[-1] - y_grids[0])/(len(y_grids)-1)
            rs = np.array([x_gap, y_gap])   # (2,)
            rs = np.repeat(rs[None], self.K, axis=0)  # (K, 2)
        else:
            if isinstance(rs, float):
                rs = rs * np.ones(self.K, 2)
            else:
                assert isinstance(rs, np.ndarray) and rs.shape == (2,)
                rs = np.repeat(rs[None], self.K, axis=0)
        assert rs.shape == (self.K, 2), rs.shape
        self.rs = torch.tensor(rs, dtype=torch.float64, device=self.device, requires_grad=train_rs)

        # (K, 2)
        vs = [[1,1] for _ in range(self.K)]
        # TODO: if train_vs, need to make vs positive
        self.vs = torch.tensor(vs, dtype=torch.float64, device=self.device, requires_grad=train_vs)

        self.kernel_distsq_gg = kernel_distsq(self.inducing_points, self.inducing_points)  # (n_gps, n_gps)

    @property
    def params(self):
        return self.log_sigmas, self.us, self.rs, self.vs

    @params.setter
    def params(self, values):
        self.log_sigmas = set_param(self.log_sigmas, values[0])
        self.us = set_param(self.us, values[1])
        self.rs = set_param(self.rs, values[2])
        self.vs = set_param(self.vs, values[3])

    def permute(self, perm):
        self.mus_init = self.mus_init[perm]
        self.log_sigmas_init = self.log_sigmas_init[perm]
        self.log_sigmas = self.log_sigmas[perm]
        self.us = self.us[perm]
        self.rs = self.rs[perm]
        self.vs = self.vs[perm]

    def log_prob(self, inputs, **kwargs):
        """

        :param inputs: (T, D)
        :param kwargs:
        :return: (T, K)
        """
        T, _ = inputs.shape

        # mean (K,D), sigma (K, D)
        p_init = Normal(self.mus_init, torch.exp(self.log_sigmas_init))  #
        log_prob_init = p_init.log_prob(inputs[0])
        assert log_prob_init.shape == (self.K, self.D), log_prob_init.shape
        log_prob_init = torch.sum(log_prob_init, dim=-1)  # (K, )

        log_prob_ar = self.log_prob_for_single_animal(inputs, **kwargs)

        assert log_prob_ar.shape == (T-1, self.K)

        log_prob = torch.cat((log_prob_init[None, ], log_prob_ar), dim=0)
        assert log_prob.shape == (T, self.K)

        return log_prob

    def log_prob_for_single_animal(self, inputs, **kwargs):
        """
        calculate the log prob for inputs[1:] based on inputs[:-1]
        :param inputs: (T, 2)
        :param kwargs:
        :return: (T-1, K)
        """

        T, d = inputs.shape
        assert d == 2, d
        # get the mu and cov based on the observations except the last one
        mu, cov = self.get_mu_and_cov_for_single_animal(inputs[:-1], **kwargs)
        # mean: (T-1, K, 2), covariance (T-1, K, 2)
        m = Normal(mu, torch.sqrt(cov))

        # evaluated the observations except the first one. (T-1, 1, 2)
        log_prob = m.log_prob(inputs[1:, None])  # (T-1, K, 2)
        log_prob = torch.sum(log_prob, dim=-1)
        assert log_prob.shape == (T-1, self.K), log_prob.shape

        return log_prob

    def get_mu_and_cov_for_single_animal(self, inputs, mu_only=False, **kwargs):
        """

        :param inputs: (T, 2)
        :param mu_only:
        :param kwargs:
        :return: mu: (T, K, 2), cov (T, K, 2)
        """

        inputs = check_and_convert_to_tensor(inputs, dtype=torch.float64, device=self.device)

        T, _ = inputs.shape

        # this is useful when train_rs =False and train_vs=False:
        Sigma = kwargs.get("Sigma", None)
        A = kwargs.get("A", None)
        if A is None:
            #print("Not using cache. Calculating Sigma, A...")
            Sigma, A = self.get_gp_cache(inputs, A_only=mu_only, **kwargs)

        A_x, A_y = A

        # (K, T, n_gps) * (K, n_gps, 1) -> (K, T, 1)
        mu_x = torch.matmul(A_x, self.us[:, :, 0:1])
        assert mu_x.shape == (self.K, T, 1)
        mu_y = torch.matmul(A_y, self.us[:, :, 1:2])
        assert mu_y.shape == (self.K, T, 1)
        mu = torch.cat((mu_x, mu_y), dim=-1)  # (K, T, 2)
        mu = torch.transpose(mu, 0, 1) # (T, K, 2)
        mu = mu + inputs[:, None]

        if mu_only:
            return mu, 0

        assert Sigma.shape == (self.K, T, 2)
        Sigma = torch.transpose(Sigma, 0, 1)

        # (K, 2)
        sigma = torch.exp(self.log_sigmas)

        # (T, K, 2)
        cov = Sigma + sigma ** 2
        return mu, cov

    def get_gp_cache(self, inputs, A_only=False, **kwargs):
        """

        :param inputs: (T, 2)
        :param A_only: return A only
        :return: Sigma (K, T, 2), A = (A_x, A_y), each is (K, T, n_gps)
        """
        T, _ = inputs.shape

        kernel_distsq_xg = kwargs.get("kernel_distsq_xg", None)
        if kernel_distsq_xg is None:
            #print("Nog using cache. Calculating kernel_distsq_xg...")
            kernel_distsq_xg = kernel_distsq(inputs, self.inducing_points)
        assert kernel_distsq_xg.shape == (T, self.n_gps), \
            "the correct size is {}, but got {}".format((T, self.n_gps), kernel_distsq_xg.shape)

        # each is (K, n_gps, n_gps)
        Kgg_x_inv, Kgg_y_inv = self.get_Kgg_inv()
        # TODO: validate using the old codes
        # (K, 1, 1) * [ (T, n_gps) / (K, 1, 1) ] -> (K, T, n_gps)
        Kxg_x = self.vs[:, 0:1, None] * torch.exp(-kernel_distsq_xg / self.rs[:, 0:1, None]**2)
        assert Kxg_x.shape == (self.K, T, self.n_gps), Kxg_x.shape
        Kxg_y = self.vs[:, 1:2, None] * torch.exp(-kernel_distsq_xg / self.rs[:, 1:2, None]**2)
        assert Kxg_y.shape == (self.K, T, self.n_gps), Kxg_y.shape

        # (K, T, n_gps) * (K, n_gps, n_gps) -> (K, T, n_gps)
        A_x = torch.matmul(Kxg_x, Kgg_x_inv)
        assert A_x.shape == (self.K, T, self.n_gps)
        A_y = torch.matmul(Kxg_y, Kgg_y_inv)
        assert A_y.shape == (self.K, T, self.n_gps)

        if A_only:
            return _, (A_x, A_y)

        kernel_distsq_xx = torch.zeros((T, 2), dtype=torch.float64, device=self.device)
        # (K, 1, 2) * [ (T, 2) / (K, 1, 2)] -> (K, T, 2)
        K_xx = self.vs[:, None] * torch.exp(-kernel_distsq_xx / self.rs[:, None] ** 2)
        assert K_xx.shape == (self.K, T, 2)

        # crossterm: K_xg * K_gg^{-1} * K_xg^T
        # (K, T, 1, n_gps) * (K, T, n_gps, 1) -> (K, T, 1, 1)
        crossterm_x = torch.matmul(A_x[:,:, None], Kxg_x[..., None])
        crossterm_y = torch.matmul(A_y[:,:, None], Kxg_y[..., None])
        crossterm = torch.cat((crossterm_x, crossterm_y), dim=-2)  #  (K, T, 2, 1)
        crossterm = torch.squeeze(crossterm, dim=-1) # (K, T, 2)
        Sigma = K_xx - crossterm
        assert Sigma.shape == (self.K, T, 2), Sigma.shape

        return Sigma, (A_x, A_y)

    def sample_x(self, z, xhist=None, transformation=False, return_np=True, **kwargs):
        """

        :param z: a scalar
        :param xhist: (T_pre, D)
        :param transformation:
        :param return_np:
        :param kwargs:
        :return: (D,)
        """
        with torch.no_grad():
            if xhist is None or len(xhist) == 0:
                mu = self.mus_init[z]  # (D,)
                if transformation:
                    sample = mu
                else:
                    sigmas_z = torch.exp(self.log_sigmas_init[z])  # (D,)
                    sample = mu + sigmas_z * torch.randn(self.D, dtype=torch.float64)  # (self.D, )
            else:
                sample = self.sample_single_animal_x(z, xhist[-1:, 0:2], transformation, **kwargs)
            assert sample.shape == (self.D, ), sample.shape

        for i in range(self.D):
            sample[i] = clip(sample[i], self.bounds[i])

        if return_np:
            sample = sample.detach().numpy()
        return sample

    def sample_single_animal_x(self, z, x_pre,expectation, **kwargs):
        """

        :param z: a scalar
        :param x_pre: (1, 2)
        :param expectation:
        :return: (2, )
        """
        assert x_pre.shape == (1, 2), x_pre.shape

        A = kwargs.get("A", None)
        Sigma = kwargs.get("Sigma", None)

        if A is None:
            #print("Not using cache. Calculating Sigma, A...")
            Sigma, A = self.get_gp_cache_condition_on_z(x_pre, z, A_only=expectation, **kwargs)

        # each is (1, n_gps)
        A_x, A_y = A

        # (1, n_gps) * (n_gps, 1) ->  (1, 1)
        mu_x = torch.matmul(A_x, self.us[z, :, 0:1])
        mu_y = torch.matmul(A_y, self.us[z, :, 1:2])
        mu = torch.cat((mu_x, mu_y), dim=-1)  # (1, 2)
        mu = torch.squeeze(mu, dim=0)
        mu = mu + x_pre[0]
        assert mu.shape == (2, )

        if expectation:
            return mu

        assert Sigma.shape == (1, 2)
        Sigma = torch.squeeze(Sigma, dim=0)

        # (2,)
        sigma = torch.exp(self.log_sigmas[z])
        cov = Sigma + sigma ** 2

        m = Normal(mu, torch.sqrt(cov))
        sample = m.sample()
        assert sample.shape == (2, )

        return sample

    def get_gp_cache_condition_on_z(self, inputs, z, A_only=False, **kwargs):
        """

        :param inputs: (T, 2)
        :param z:
        :return: Sigma (T, 2) A = (A_x, A_y), each is (T, n_gps)
        """
        T, _ = inputs.shape

        kernel_distsq_xg = kwargs.get("kernel_distsq_xg", None)

        if kernel_distsq_xg is None:
            #print("Nog using cache. Calculating kernel_distsq_xg...")
            kernel_distsq_xg = kernel_distsq(inputs, self.inducing_points)
        assert kernel_distsq_xg.shape == (T, self.n_gps)

        Kgg_inv = kwargs.get("Kgg_inv", None)
        if Kgg_inv:
            Kgg_x_inv = Kgg_inv[0][z]
            Kgg_y_inv = Kgg_inv[1][z]
        else:
            Kgg_x_inv, Kgg_y_inv = self.get_Kgg_inv_condition_on_z(z)

        Kxg_x = self.vs[z, 0] * torch.exp(-kernel_distsq_xg / self.rs[z, 0]**2)
        assert Kxg_x.shape == (T, self.n_gps)
        Kxg_y = self.vs[z, 1] * torch.exp(-kernel_distsq_xg / self.rs[z, 1]**2)
        assert Kxg_y.shape == (T, self.n_gps)
        # (T, n_gps) * (n_gps, n_gps)
        A_x = torch.matmul(Kxg_x, Kgg_x_inv)
        A_y = torch.matmul(Kxg_y, Kgg_y_inv)
        assert A_x.shape == A_y.shape == (T, self.n_gps), "{}, {}, {}".format((T, self.n_gps), A_x.shape, A_y.shape)

        if A_only:
            return 0, (A_x, A_y)

        kernel_distsq_xx = torch.zeros((T, 2), dtype=torch.float64, device=self.device)
        K_xx = self.vs[z] * torch.exp(-(kernel_distsq_xx / self.rs[z] ** 2))
        assert K_xx.shape == (T, 2)

        # crossterm: K_xg * K_gg^{-1} * K_xg^T
        # (T, 1, n_gps) * (T, n_gps, 1) -> (T, 1, 1)
        crossterm_x = torch.matmul(A_x[:, None], Kxg_x[..., None])
        crossterm_y = torch.matmul(A_y[:, None], Kxg_y[..., None])
        crossterm = torch.cat((crossterm_x, crossterm_y), dim=-2)  # (T, 2, 1)
        crossterm = torch.squeeze(crossterm, dim=-1)  # (T, 2)
        Sigma = K_xx - crossterm

        assert Sigma.shape == (T, 2), Sigma.shape

        return Sigma, (A_x, A_y)

    def get_Kgg_inv(self):
        # (K, 1, 1) * [(n_gps, n_gps) / (K, 1, 1)
        K_gg_x = self.vs[:, 0:1, None] * torch.exp(-self.kernel_distsq_gg / self.rs[:, 0:1, None]**2)
        K_gg_y = self.vs[:, 1:2, None] * torch.exp(-self.kernel_distsq_gg / self.rs[:, 1:2, None]**2)

        assert K_gg_x.shape == (self.K, self.n_gps, self.n_gps)
        assert K_gg_y.shape == (self.K, self.n_gps, self.n_gps)

        K_gg_x_inv = torch.inverse(K_gg_x)
        K_gg_y_inv = torch.inverse(K_gg_y)

        return K_gg_x_inv, K_gg_y_inv

    def get_Kgg_inv_condition_on_z(self, z):
        Kgg_x = self.vs[z, 0] * torch.exp(-self.kernel_distsq_gg / self.rs[z, 0] ** 2)
        Kgg_y = self.vs[z, 1] * torch.exp(-self.kernel_distsq_gg / self.rs[z, 1] ** 2)

        assert Kgg_x.shape == (self.n_gps, self.n_gps)
        assert Kgg_y.shape == (self.n_gps, self.n_gps)

        Kgg_x_inv = torch.inverse(Kgg_x)
        Kgg_y_inv = torch.inverse(Kgg_y)

        return Kgg_x_inv, Kgg_y_inv


