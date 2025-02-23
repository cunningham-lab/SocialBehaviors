import torch
import torch.nn as nn

import numpy as np

from ssm_ptc.observations.base_observation import BaseObservation
from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.distributions.base_distribution import BaseDistribution
from ssm_ptc.distributions.truncatednormal import TruncatedNormal
from ssm_ptc.utils import check_and_convert_to_tensor, set_param


class ARTruncatedNormalObservation(BaseObservation):
    """
    A mixture of distributions
    """

    def __init__(self, K, D, M=0, transformation='linear', lags=1, mus_init=None, sigmas=None,
                 bounds=None, train_sigma=True):
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
        super(ARTruncatedNormalObservation, self).__init__(K, D, M)

        if isinstance(transformation, str):
            if transformation == 'linear':
                self.transformation = LinearTransformation(K=self.K, D=self.D, lags=lags)
                self.lags = lags
        else:
            assert isinstance(transformation, BaseTransformation)
            self.transformation = transformation
            self.lags = self.transformation.lags

        # consider diagonal covariance
        if sigmas is None:
            log_sigmas = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64)
        else:
            # TODO: assert sigmas positive
            assert sigmas.shape == (self.K, self.D)
            log_sigmas = check_and_convert_to_tensor(np.log(sigmas), dtype=torch.float64)
        self.log_sigmas = nn.Parameter(log_sigmas, requires_grad=train_sigma)

        if bounds is None:
            raise ValueError("Please provide bounds.")
            # default bound for each dimension is [0,1]
            #self.bounds = torch.cat((torch.zeros(self.D, dtype=torch.float64)[:, None],
            #                         torch.ones(self.D, dtype=torch.float64)[:, None]), dim=1)
        else:
            self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64)
            assert self.bounds.shape == (self.D, 2)

        if mus_init is None:
            self.mus_init = torch.eye(self.K, self.D, dtype=torch.float64)
        else:
            self.mus_init = torch.tensor(mus_init, dtype=torch.float64)

        # consider diagonal covariance
        self.log_sigmas_init = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64)

    def _compute_mus_based_on(self, data):
        # not tested.
        # TODO: test this method
        mus = self.transformation.transform(data)
        return data

    def _compute_mus_for(self, data, **kwargs):
        """
        compute the mean vector for each observation (using the previous observation, or mus_init)
        :param data: (T,D)
        :return: mus: (T, K, D)
        """
        T, D = data.shape
        assert D == self.D

        if T < self.lags:
            mus = self.mus_init * torch.ones(T, self.K, self.D, dtype=torch.float64)
        else:
            mus_rest = self.transformation.transform(data[:-1], **kwargs)  # (T-momentum_lags, K, D)
            assert mus_rest.shape == (T-1-self.lags+1, self.K, D)

            mus = torch.cat((self.mus_init * torch.ones(self.lags, self.K, self.D, dtype=torch.float64), mus_rest))

        assert mus.shape == (T, self.K, self.D)
        return mus

    def log_prob(self, data, **kwargs):
        """

        :param data: shape (T, D)
        :return: log prob under each possible z_t: shape (T, K)
        """

        mus = self._compute_mus_for(data, **kwargs)  # (T, K, D)
        T = data.shape[0]

        p_init = TruncatedNormal(mus=mus[0], log_sigmas=self.log_sigmas_init, bounds=self.bounds)  # mus[0] (K, D)
        log_prob_init = p_init.log_prob(data[0])  # data[0] (D, ). log_prob_init: (K, D)
        log_prob_init = torch.sum(log_prob_init, dim=-1)  # (K, )

        if T == 1:
            return log_prob_init[None,]

        dist = TruncatedNormal(mus=mus[1:], log_sigmas=self.log_sigmas, bounds=self.bounds)

        log_prob_ar = dist.log_prob(data[1:, None])  # (T-1, K, D)
        log_prob_ar = torch.sum(log_prob_ar, dim=-1)  # (T-1, K)

        return torch.cat((log_prob_init[None,], log_prob_ar))

    def sample_x(self, z, xhist=None, with_noise=False, return_np=True):
        """

        :param z: ()
        :param xhist: (T_pre, D)
        :param return_np:
        :return: x: shape (D, )
        """

        # currently only support non-reparameterizable rejection sampling

        # no previous x
        if xhist is None or xhist.shape[0] < self.lags:
            mu = self.mus_init[z]  # (D,)
        else:
            # sample from the autoregressive distribution
            assert len(xhist.shape) == 2
            mu = self.transformation.transform_condition_on_z(z, xhist[-self.lags:])  # (D, )
            assert mu.shape == (self.D,)

        if with_noise:
            samples = mu
            # some ad-hoc way to address bound issue
            for d in range(self.D):
                if samples[d] <= self.bounds[d, 0]:
                    samples[d] = self.bounds[d, 0] + 0.1 * torch.rand(1, dtype=torch.float64)
                elif samples[d] >= self.bounds[d, 1]:
                    samples[d] = self.bounds[d, 1] - 0.1 * torch.rand(1, dtype=torch.float64)

            samples = samples.detach()

        else:
            dist = TruncatedNormal(mus=mu, log_sigmas=self.log_sigmas[z], bounds=self.bounds)
            samples = dist.sample()
        if return_np:
            return samples.numpy()
        return samples

    def rsample_x(self, z, xhist=None, with_noise=False):
        """
        generate reparameterized samples
        :param z: shape ()
        :param xhist: shape (T_pre, D)
        :return: x: shape (D,)
        """
       # TODO: add rejection sampling
        raise NotImplementedError















