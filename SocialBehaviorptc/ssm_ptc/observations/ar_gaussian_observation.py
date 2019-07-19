import torch
from torch.distributions import Normal, MultivariateNormal

import numpy as np
import numpy.random as npr

from ssm_ptc.observations.base_observation import BaseObservations
from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.transformations.linear import LinearTransformation


class ARGaussianObservation(BaseObservations):
    """
    A mixture of gaussians
    # TODO: subclassing ARObservation
    """

    def __init__(self, K, D, M, transformation, mus_init=None, sigmas=None, lags=1):
        super(ARGaussianObservation, self).__init__(K, D, M)

        if mus_init is None:
            self.mus_init = torch.zeros(self.K, self.D, dtype=torch.float64)
        else:
            self.mus_init = torch.tensor(mus_init, dtype=torch.float64)

        # consider diagonal covariance
        self.log_sigmas_init = torch.tensor(np.log(5*np.ones((K, D))), dtype=torch.float64)

        if sigmas is None:
            self.log_sigmas = torch.tensor(np.log(5*np.ones((K, D))), dtype=torch.float64, requires_grad=True)
        else:
            # TODO: assert sigmas positive
            assert sigmas.shape == (self.K, self.D)
            self.log_sigmas = torch.tensor(np.log(sigmas), dtype=torch.float64, requires_grad=True)

        self.lags = lags

        if isinstance(transformation, str):
            if transformation == 'linear':
                self.transformation = LinearTransformation(K=self.K, D=self.D, lags=self.lags)
        else:
            assert isinstance(transformation, BaseTransformation)
            self.transformation = transformation
            self.lags = self.transformation.lags

    @property
    def params(self):
        # do not train initial parameters
        return [self.log_sigmas] + self.transformation.params

    def permute(self, perm):
        self.mus_init = self.mus_init[perm]
        self.log_sigmas = self.log_sigmas[perm]
        self.transformation.permute(perm)

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
        # TODO: test lags
        T, D = data.shape
        assert D == self.D

        mus_rest = self.transformation.transform(data[:-1])  # (T-1-lags+1, K, D)
        assert mus_rest.shape == (T-1-self.lags+1, self.K, D)

        # add repeated lags
        mus = torch.cat((self.mus_init * torch.ones(self.lags, self.K, self.D, dtype=torch.float64), mus_rest))

        assert mus.shape == (T, self.K, self.D)
        return mus

    def log_prob(self, data):
        """

        :param data: shape (T, D)
        :return: log prob under each possible z_t: shape (T, K)
        """

        mus = self._compute_mus_for(data)  # (T, K, D)
        T = mus.shape[0]

        p_init = Normal(mus[0], torch.exp(self.log_sigmas_init))  # mus[0] (K, D)
        log_prob_init = p_init.log_prob(data[0])  # data[0] (D). log_prob_init: (K, D)
        log_prob_init = torch.sum(log_prob_init, dim=-1)  # (K, )

        p = Normal(mus[1:], torch.exp(self.log_sigmas))

        log_prob_ar = p.log_prob(data[1:,None]) # (T, K, D)
        log_prob_ar = torch.sum(log_prob_ar, dim=-1)  # (T-1, K)

        return torch.cat((log_prob_init[None,], log_prob_ar))

    def sample_x(self, z, xhist=None, return_np=True):
        """
        generate samples
        """

        with torch.no_grad():
            x = self.rsample_x(z, xhist)
        if return_np:
            return x.numpy()
        return x

    def rsample_x(self, z, xhist=None):
        """
        generate reparameterized samples
        :param z: shape ()
        :param xhist: shape (T_pre, D)
        :return: x: shape (D,)
        """
        # TODO: test lags
        # no previous x
        if xhist is None or xhist.shape[0] < self.lags:
            mu = self.mus_init[z]  # (D,)
            sigmas_z = torch.exp(self.log_sigmas_init[z])  # (D,)
        else:
            # sample from the autoregressive distribution
            assert len(xhist.shape) == 2
            mu = self.transformation.transform_condition_on_z(z, xhist[-self.lags:])  # (D, )
            sigmas_z = torch.exp(self.log_sigmas[z])  # (D,)

        out = mu + sigmas_z * torch.randn(self.D, dtype=torch.float64)  # (self.D, )

        return out













