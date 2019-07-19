import torch
from torch.distributions import Normal, MultivariateNormal

import numpy as np
import numpy.random as npr

from ssm_ptc.observations.base_observation import BaseObservations
from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.distributions.base_distribution import BaseDistribution
from ssm_ptc.distributions.sigmoidnormal import SigmoidNormal
from ssm_ptc.utils import check_and_convert_to_tensor


class ARSigmoidNormalObservation(BaseObservations):
    """
    A mixture of distributions
    """

    def __init__(self, K, D, M, transformation, mus_init=None, sigmas=None, bounds=None, alpha=0.2, train_sigma=True,
                 lags=1):
        """
        x ~ N(mu, sigma)
        h_1 = sigmoid(h)
        x_tran = (upperbound - lowerbound) * h_1 + lowerbound

        :param K: number of hidden states
        :param D: dimension of the observation
        :param M: dimension of the input
        :param transformation: used for auto-regressive relationship. transforms the previous observations into
         parameters of the emission distribution
        :param mus_init: (K, D)
        :param sigmas: (K, D) -- recover the diagonal covariance of shape (K,D,D)
        :param bounds: (D, 2) -- lower and upper bounds for each dimension of the observation
        """
        super(ARSigmoidNormalObservation, self).__init__(K, D, M)

        if isinstance(transformation, str):
            if transformation == 'linear':
                self.transformation = LinearTransformation(K=self.K, d_in=self.D, d_out=self.D)
        else:
            assert isinstance(transformation, BaseTransformation)
            self.transformation = transformation

        # consider diagonal covariance
        if sigmas is None:
            self.log_sigmas = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, requires_grad=train_sigma)
        else:
            # TODO: assert sigmas positive
            assert sigmas.shape == (self.K, self.D)
            self.log_sigmas = torch.tensor(np.log(sigmas), dtype=torch.float64, requires_grad=train_sigma)

        if bounds is None:
            self.bounds = torch.ones(self.D, 2)
        else:
            self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64)
            assert self.bounds.shape == (self.D, 2)

        if mus_init is None:
            self.mus_init = torch.eye(self.K, self.D, dtype=torch.float64, requires_grad=True)
        else:
            self.mus_init = torch.tensor(mus_init, dtype=torch.float64, requires_grad=True)

        self.alpha = alpha

        self.lags = lags

    @property
    def params(self):
        return [self.mus_init, self.log_sigmas] + self.transformation.params
        #return [self.mus_init] + self.transformation.params

    def permute(self, perm):
        self.mus_init = self.mus_init[perm]
        self.log_sigmas = self.log_sigmas[perm]
        self.transformation.permute(perm)

    def _compute_mus_based_on(self, data):
        # not tested. TODO: test this method
        mus = self.transformation.transform(data)
        return data

    def _compute_mus_for(self, data):
        """
        compute the mean vector for each observation (using the previous observation, or mus_init)
        :param data: (T,D)
        :return: mus: (T, K, D)
        """
        T, D = data.shape
        assert D == self.D

        # TODO: add lags  (T-lags, K, D)
        mus_rest = self.transformation.transform(data[:-1])  # (T-lags, K, D)
        assert mus_rest.shape == (T-self.lags, self.K, D)

        mus = torch.cat((self.mus_init[None, ], mus_rest))

        assert mus.shape == (T, self.K, self.D)
        return mus

    def log_prob(self, data):
        """

        :param data: shape (T, D)
        :return: log prob under each possible z_t: shape (T, K)
        """

        mus = self._compute_mus_for(data)  # (T, K, D)

        p = SigmoidNormal(mus=mus, log_sigmas=self.log_sigmas, bounds=self.bounds, alpha=self.alpha)

        out = p.log_prob(data[:, None])  # (T, K, D)
        out = torch.sum(out, dim=-1)  # (T, K)
        return out

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

        # no previous x
        if xhist is None or xhist.shape[0] == 0:
            mu = self.mus_init[z]  # (D,)
        else:
            # sample from the autoregressive distribution
            # currently consider lag = 1
            assert len(xhist.shape) == 2
            x_pre = xhist[-1:]  # (1, D)
            mu = self.transformation.transform_condition_on_z(z, x_pre)  # (1, D_out)
            assert mu.shape == (1, self.D)
            mu = torch.squeeze(mu, 0)  # (D, )

        p = SigmoidNormal(mus=mu, log_sigmas=self.log_sigmas[z], bounds=self.bounds, alpha=self.alpha)

        out = p.sample()
        assert out.shape == (self.D, )

        return out













