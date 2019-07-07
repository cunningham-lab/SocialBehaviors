import torch
from torch.distributions import Normal, MultivariateNormal

import numpy as np
import numpy.random as npr

from socialbehavior.observations.base_observation import BaseObservations
from socialbehavior.transformations.base_transformation import BaseTransformation
from socialbehavior.transformations.linear import LinearTransformation


class ARGaussianObservation(BaseObservations):
    """
    A mixture of gaussians
    """

    def __init__(self, K, D, M, transformation, mus_init=None, sigmas=None):
        super(ARGaussianObservation, self).__init__(K, D, M)

        if mus_init is None:
            self.mus_init = torch.eye(self.K, self.D, dtype=torch.float64, requires_grad=True)
        else:
            self.mus_init = torch.tensor(mus_init, dtype=torch.float64, requires_grad=True)

        # consider diagonal covariance
        if sigmas is None:
            self.log_sigmas = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, requires_grad=True)
        else:
            # TODO: assert sigmas positive
            assert sigmas.shape == (self.K, self.D)
            self.log_sigmas = torch.tensor(np.log(sigmas), dtype=torch.float64, requires_grad=True)

        if isinstance(transformation, str):
            if transformation == 'linear':
                self.transformation = LinearTransformation(K=self.K, d_in=self.D, d_out=self.D)
        else:
            assert isinstance(transformation, BaseTransformation)
            self.transformation = transformation

    @property
    def params(self):
        return [self.mus_init, self.log_sigmas] + self.transformation.params

    def _get_scale_tril(self, log_sigmas):
        sigmas = torch.exp(log_sigmas)
        return torch.diag_embed(sigmas)

    def _compute_mus_based_on(self, data):
        pass

    def _compute_mus_for(self, data):
        """
        compute the mean vector for each observation (using the previous observation, or mus_init)
        :param data: (T,D)
        :return: mus: (T, K, D)
        """
        T, D = data.shape
        assert D == self.D

        mus_rest = self.transformation.transform(data[:-1])  # (T-1, K, D)
        assert mus_rest.shape == (T-1, self.K, D)

        mus = torch.cat((self.mus_init[None, ], mus_rest))

        assert mus.shape == (T, self.K, self.D)
        return mus

    def log_prob(self, data):
        """

        :param data: shape (T, D)
        :return: log prob under each possible z_t: shape (T, K)
        """

        mus = self._compute_mus_for(data)  # (T, K, D)

        #cov = self._get_cov(self.log_sigma_sq)
        #assert cov.shape == (self.K, self.D, self.D)

        p = Normal(mus, torch.exp(self.log_sigmas))

        out = p.log_prob(data[:,None]) # (T, K, D)
        out = torch.sum(out, dim=-1)
        return out

    def sample_x(self, z, xhist):
        """
        generate samples
        """

        with torch.no_grad():
            return self.rsample_x(z, xhist)

    def rsample_x(self, z, xhist):
        """
        generate reparameterized samples
        :param z: shape ()
        :param xhist: shape (T_pre, D)
        :return: x: shape (D,)
        """

        sigmas_z = torch.exp(self.log_sigmas[z])  # (D,)
        assert sigmas_z.shape == (self.D,)

        # no previous x
        if xhist.shape[0] == 0:
            mu = self.mus_init[z]  # (D,)
        else:
            # sample from the autoregressive distribution
            # currently consider lag = 1
            x_pre = xhist[-1:]  # (1, D)
            mu = self.transformation.transform_condition_on_z(z, x_pre)  # (1, D_out)
            assert mu.shape == (1, self.D)
            mu = torch.squeeze(mu, 0)  # (D, )

        out = mu + sigmas_z * torch.randn(self.D, dtype=torch.float64)  # (self.D, )

        return out












