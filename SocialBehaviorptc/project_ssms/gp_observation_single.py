import torch
from torch.distributions import Normal, MultivariateNormal
import numpy as np

from project_ssms.gp_observation import kernel_distsq, batch_kernle_dist_sq
from project_ssms.utils import clip
from ssm_ptc.utils import check_and_convert_to_tensor, set_param, get_np
from ssm_ptc.observations.base_observation import BaseObservation



class GPObservationSingle(BaseObservation):
    """
    Learnable parameters: weights of each grid vertex
    weights of any random point = a weighted combination of weights of all the grid vertices
    where the weights = softmax
    """
    def __init__(self, K, D, x_grids, y_grids, bounds, mus_init=None,
                rs=None, train_rs=False, train_vs=False,
                 device=torch.device('cpu')):
        assert D == 2
        super(GPObservationSingle, self).__init__(K, D)

        self.device = device

        self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64, device=self.device)
        assert self.bounds.shape == (self.D, 2), self.bounds.shape

        # specify distribution parameters
        if mus_init is None:
            self.mus_init = torch.zeros(self.K, self.D, dtype=torch.float64, device=self.device)
        else:
            self.mus_init = check_and_convert_to_tensor(mus_init, dtype=torch.float64, device=self.device)
        # consider diagonal covariance
        self.log_sigmas_init = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, device=self.device)
        # shape (K,D)
        self.log_sigmas = torch.tensor(np.log(5*np.ones((K, D))), dtype=torch.float64, device=self.device,
                                         requires_grad=True)

        # specify gp dynamics parameters
        self.x_grids = check_and_convert_to_tensor(x_grids, dtype=torch.float64, device=self.device)  # [x_0, x_1, ..., x_m]
        self.y_grids = check_and_convert_to_tensor(y_grids, dtype=torch.float64, device=self.device)  # a list [y_0, y_1, ..., y_n]

        self.inducing_points = torch.tensor([(x_grid, y_grid) for x_grid in self.x_grids for y_grid in self.y_grids],
                                            device=self.device)  # (n_gps, 2)
        self.n_gps = self.inducing_points.shape[0]

        self.us = torch.rand(self.K, self.n_gps, self.D, dtype=torch.float64, device=self.device, requires_grad=True)

        # define GP parameters, suppose parameters work for all Ks
        if rs is None:
            x_grids = get_np(x_grids)
            y_grids = get_np(y_grids)
            x_gap = (x_grids[-1] - x_grids[0])/(len(x_grids)-1)
            y_gap = (y_grids[-1] - y_grids[0])/(len(y_grids)-1)
            xy_gap = np.sqrt(x_gap**2 + y_gap**2)
            rs = np.array([[x_gap, xy_gap], [xy_gap, y_gap]])
            rs = np.repeat(rs[None], self.K, axis=0)
        elif isinstance(rs, float):
            rs = rs * np.ones(self.K, 2, 2)
            assert rs.shape == (self.K, 2, 2), rs.shape
        self.rs = torch.tensor(rs, dtype=torch.float64, device=self.device, requires_grad=train_rs)

        # (K, 2,2)
        vs = [[[1,0],[0,1]] for _ in range(self.K)]
        self.vs = torch.tensor(vs, dtype=torch.float64, device=self.device, requires_grad=train_vs)
        # real_vs = vs**2

        self.kernel_distsq_gg = kernel_distsq(self.inducing_points, self.inducing_points)  # (n_gps*2, n_gps*2)

    @property
    def params(self):
        return self.us, self.rs, self.vs

    @params.setter
    def params(self, values):
        self.us = set_param(self.us, values[0])
        self.rs = set_param(self.rs, values[1])
        self.vs = set_param(self.vs, values[2])

    def permute(self, perm):
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
        # mean: (T-1, K, 2), covariance (T-1, K, 2, 2)
        m = MultivariateNormal(mu, cov)

        # evaluated the observations except the first one. (T-1, 1, 2)
        log_prob = m.log_prob(inputs[1:, None])
        assert log_prob.shape == (T-1, self.K)

        return log_prob


    def get_mu_and_cov_for_single_animal(self, inputs, mu_only=False, **kwargs):

        inputs = check_and_convert_to_tensor(inputs, dtype=torch.float64, device=self.device)

        T, _ = inputs.shape

        # this is useful when train_rs =False and train_vs=False:
        Sigma = kwargs.get("Sigma", None)
        A = kwargs.get("A", None)
        if A is None:
            #print("Not using cache. Calculating Sigma, A...")
            Sigma, A = self.get_gp_cache(inputs, A_only=mu_only, **kwargs)
        assert A.shape == (T, self.K, 2, 2 * self.n_gps), A.shape

        # calculate the dynamics at the grids
        us = torch.reshape(self.us, (self.K, -1))  # (K, n_gps*2)

        # (T-1, K, 2, 2*n_gps) * (K, 2*n_gps, 1) ->  (T-1, K, 2, 1)
        mu = torch.matmul(A, us[..., None])
        mu = torch.squeeze(mu, dim=-1)
        mu = mu + inputs[:, None]
        assert mu.shape == (T, self.K, 2)

        if mu_only:
            return mu, 0

        assert Sigma.shape == (T, self.K, 2, 2), Sigma.shape

        # (K, 2)
        sigma = torch.exp(self.log_sigmas)
        # (K, 2, 2)
        sigma = torch.diag_embed(sigma)

        cov = Sigma + sigma
        return mu, cov

    def get_mu(self, inputs, **kwargs):
        T, D = inputs.shape
        assert D == 2, D
        _, mu = self.get_mu_and_cov_for_single_animal(inputs[:, 0:2], mu_only=True, **kwargs)
        assert mu.shape == (T, self.K, self.D)
        return mu

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

        assert A.shape == (1, 2, self.n_gps*2)
        A = torch.squeeze(A, dim=0)

        u = self.us[z]
        assert u.shape == (self.n_gps, 2), \
            "the correct shape is {}, instead we got {}".format((self.n_gps, 2), u.shape)
        u = torch.reshape(u, (-1,))  # (n_gps*2, )

        # (2, n_gps*2) * (n_gps*2, 1) ->  (2, 1)
        mu = torch.matmul(A, u[..., None])
        assert mu.shape == (2, 1), mu.shape
        mu = torch.squeeze(mu, dim=-1)  # (2, )
        mu = mu + x_pre[0]
        assert mu.shape == (2, )

        if expectation:
            return mu

        assert Sigma.shape == (1, 2, 2)
        Sigma = torch.squeeze(Sigma, dim=0)

        # (2,)
        sigma = torch.exp(self.log_sigmas[z])
        assert sigma.shape == (2, ), sigma.shape
        sigma =  torch.diag(sigma)
        assert sigma.shape == (2,2), sigma.shape

        cov = Sigma + sigma

        m = MultivariateNormal(mu, cov)
        sample = m.sample()
        assert sample.shape == (2, )

        return sample

    def get_gp_cache_condition_on_z(self, inputs, z, A_only=False, **kwargs):
        """

        :param inputs: (T, 2)
        :param z:
        :return: Sigma (T, 2, 2,) A (T, 2, n_gps*2)
        """
        T, _ = inputs.shape

        kernel_distsq_xg = kwargs.get("kernel_distsq_xg", None)

        if kernel_distsq_xg is None:
            #print("Nog using cache. Calculating kernel_distsq_xg...")
            kernel_distsq_xg = kernel_distsq(inputs, self.inducing_points)
        assert kernel_distsq_xg.shape == (T*2, self.n_gps*2)

        Kgg_inv = self.get_Kgg_inv()
        rs = self.rs[z]  # (2,2)
        vs = self.vs[z]  # (2, 2)

        repeated_rs = rs.repeat(T, self.n_gps)
        repeated_vs = vs.repeat(T, self.n_gps)
        K_xg = repeated_vs * torch.exp(-(kernel_distsq_xg / repeated_rs**2))
        assert K_xg.shape == (T * 2, self.n_gps * 2)
        K_xg = torch.reshape(K_xg, (T, 2, self.n_gps * 2))

        # (T, 2, n_gps*2) * (n_gps*2, n_gps*2) -> (T, 2, n_gps*2)
        A = torch.matmul(K_xg, Kgg_inv[z])
        assert A.shape == (T, 2, self.n_gps * 2)

        if A_only:
            return 0, A

        kernel_distsq_xx = kwargs.get("kernel_distsq_xx", None)
        if kernel_distsq_xx is None:
            #print("Not using cache. Calculating kernel_distsq_xx...")
            kernel_distsq_xx = batch_kernle_dist_sq(inputs)
        assert kernel_distsq_xx.shape == (T, 2, 2)

        K_xx = vs * torch.exp(-(kernel_distsq_xx / rs ** 2))
        assert K_xx.shape == (T, 2, 2)

        Sigma = K_xx - torch.matmul(A, torch.transpose(K_xg, -2, -1))
        assert Sigma.shape == (T, 2, 2), Sigma.shape

        # vs shape (K, 2,2,2)
        Sigma = self.vs[z]**2 * Sigma
        return Sigma, A

    def get_gp_cache(self, inputs, A_only=False, **kwargs):
        """

        :param inputs: (T, 2)
        :param A_only: return A only
        :return: Sigma (T, K, 2, 2), A (T, K, 2, 2*n_gps)
        """
        T, _ = inputs.shape

        kernel_distsq_xg = kwargs.get("kernel_distsq_xg", None)
        if kernel_distsq_xg is None:
            #print("Nog using cache. Calculating kernel_distsq_xg...")
            kernel_distsq_xg = kernel_distsq(inputs, self.inducing_points)
        assert kernel_distsq_xg.shape == (T*2, self.n_gps*2), \
            "the correct size is {}, but got {}".format((T*2, self.n_gps*2), kernel_distsq_xg.shape)

        # (K, n_gps*2, n_gps*2)
        Kgg_inv = self.get_Kgg_inv()

        repeated_rs = self.rs.repeat(1, T, self.n_gps)
        repeated_vs = self.vs.repeat(1, T, self.n_gps)
        K_xg = repeated_vs * torch.exp(-(kernel_distsq_xg / repeated_rs**2))
        assert K_xg.shape == (self.K, T*2, self.n_gps*2)
        K_xg = torch.reshape(K_xg, (self.K, T, 2, self.n_gps*2))
        K_xg = torch.transpose(K_xg, 0, 1)  # (T, K, 2, n_gps*2)

        # (T, K, 2, n_gps*2) * (K, n_gps*2, n_gps*2) -> (T, K, 2, n_gps*2)
        A = torch.matmul(K_xg, Kgg_inv)
        assert A.shape == (T, self.K, 2, self.n_gps*2)

        if A_only:
            return 0, A

        kernel_distsq_xx = kwargs.get("kernel_distsq_xx", None)
        if kernel_distsq_xx is None:
            kernel_distsq_xx = batch_kernle_dist_sq(inputs)
        assert kernel_distsq_xx.shape == (T, 2, 2)

        K_xx = self.vs * torch.exp(-(kernel_distsq_xx[:, None] / self.rs ** 2))
        assert K_xx.shape == (T, self.K, 2, 2)

        Sigma = K_xx - torch.matmul(A, torch.transpose(K_xg, -2,-1))
        assert Sigma.shape == (T, self.K, 2, 2), Sigma.shape

        Sigma = self.vs**2 * Sigma
        return Sigma, A

    def get_Kgg_inv(self):

        repeated_rs = self.rs.repeat(1, self.n_gps, self.n_gps)
        assert repeated_rs.shape == (self.K, self.n_gps*2, self.n_gps*2), \
            "the correct shape is {}, but got {}".format((self.K, self.n_gps*2, self.n_gps*2), repeated_rs.shape)

        repeated_vs = self.vs.repeat(1, self.n_gps, self.n_gps)

        K_gg = repeated_vs * torch.exp(-self.kernel_distsq_gg / (repeated_rs ** 2))
        assert K_gg.shape == (self.K, self.n_gps*2, self.n_gps*2)

        K_gg_inv = torch.inverse(K_gg)

        return K_gg_inv


