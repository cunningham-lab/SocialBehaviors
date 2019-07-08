"""
the code is mainly based on https://github.com/slinderman/ssm
"""

import torch
import numpy as np
import numpy.random as npr

from socialbehavior.observations.base_observation import BaseObservations
from socialbehavior.observations.ar_gaussian_observation import ARGaussianObservation

from socialbehavior.message_passing.primitives import viterbi
from socialbehavior.message_passing.normalizer import hmmnorm_cython

from socialbehavior.utils import check_and_convert_to_tensor


class HMM:

    def __init__(self, K, D, M=0, observation="gaussian"):
        """

        :param K: number of hidden states
        :param D: dimension of observations
        :param M: dimension of inputs
        """
        assert isinstance(K, int)
        assert isinstance(D, int)
        assert isinstance(M, int)

        self.K = K
        self.D = D
        self.M = M

        # unnormalized initial and transition probability
        self.pi0 = torch.ones(self.K, dtype=torch.float64, requires_grad=True)
        self.P = torch.ones(self.K, self.K, dtype=torch.float64, requires_grad=True)

        if isinstance(observation, str):
            if observation == "gaussian":
                self.observation = ARGaussianObservation(self.K, self.D, self.M, transformation='linear')
        elif isinstance(observation, BaseObservations):
            self.observation = observation
        else:
            raise Exception("Invalid observation type.")

    def sample(self, T, prefix=None, return_np=True):
        """
        Sample synthetic data form from thhe model.
        :param T: int, the number of time steps to sample
        :param prefix: (z_pre, x_pre), preceding hidden states and observations.
        z_pre: shape (T_pre,)
        x_pre: shape (T_pre, D)
        :return: (z_sample, x_sample)
        z_sample: shape (T,)
        x_sample: shape (T, D)
        """
        K = self.K
        D = self.D
        M = self.M

        # get dtype of the observations
        dummy_data = self.observation.sample_x(0, np.empty(0,), return_np=False)
        dtype = dummy_data.dtype

        if prefix is None:
            # no prefix is given. Sample the initial state as the prefix
            T_pre = 1
            z = torch.empty(T, dtype=torch.int)
            data = torch.empty((T, D), dtype=dtype)

            # sample the first state from the initial distribution
            pi0 = torch.nn.Softmax(dim=0)(self.pi0.detach())
            z[0] = npr.choice(self.K, p=pi0)
            data[0] = self.observation.sample_x(z[0], data[:0], return_np=False)

            # We only need to sample T-1 datapoints now
            T = T - 1
        else:
            # check that the prefix is of the right shape
            z_pre, x_pre = prefix
            assert len(z_pre.shape) == 1
            T_pre = z_pre.shape[0]
            assert x_pre.shape == (T_pre, self.D)

            # construct thhe states and data
            z = torch.cat((z_pre, torch.empty(T, dtype=torch.int)))
            assert z.shape == (T_pre + T, )
            data = torch.cat((x_pre, torch.empty((T, D), dtype=dtype)))

        P = torch.nn.Softmax(dim=1)(self.P.detach()) # (K, K)
        for t in range(T_pre, T_pre + T):
            z[t] = npr.choice(K, p=P[z[t-1]])

            data[t] = self.observation.sample_x(z[t], data[:t], return_np=False)

        assert z.requires_grad is False
        assert data.requires_grad is False

        if prefix is None:
            if return_np:
                return z.numpy(), data.numpy()
            return z, data
        else:
            if return_np:
                return z[T_pre:].numpy(), data[T_pre:].numpy()
            return z[T_pre:], data[T_pre:]

    def loss(self, data):
        return -1. * self.log_likelihood(data)

    def log_likelihood(self, data):
        """

        :param data : x, shape (T, D)
        :return: log p(x)
        """
        T = data.shape[0]
        log_pi0 = torch.nn.LogSoftmax(dim=0)(self.pi0)  # (K, )
        log_P = torch.nn.LogSoftmax(dim=1)(self.P)  # (K, K)

        if T == 1:
            log_Ps = log_P[None,][:0]
        else:
            log_Ps = log_P[None,].repeat(T-1, 1, 1)  # (T-1, K, K)

        ll = self.observation.log_prob(data)  # (T, K)

        return hmmnorm_cython(log_pi0, log_Ps, ll)

    @property
    def params(self):
        """
        :return: pi0, P, mus_init, log_sigmas, As, bs ...
        """
        return [self.pi0, self.P] + self.observation.params

    @property
    def params_require_grad(self):
        # TODO: add this
        return 0

    # numpy operation
    def most_likely_states(self, data):
        log_pi0 = torch.nn.LogSoftmax(dim=0)(self.pi0).detach().numpy()  # (K, )
        log_Ps = torch.nn.LogSoftmax(dim=1)(self.P).detach().numpy()  # (K, K)

        log_likes = self.observation.log_prob(data).detach().numpy()
        return viterbi(log_pi0, log_Ps[None,], log_likes)

    def permute(self, perm):
        self.pi0 = self.pi0[perm]
        self.P = self.P[np.ix_(perm, perm)]
        self.observation.permute(perm)

    # return np
    def sample_condition_on_zs(self, zs, x0=None, return_np=True):
        """

        :param zs:
        :param x0: shape (D,)
        :return:
        """

        zs = check_and_convert_to_tensor(zs, dtype=torch.int)
        T = zs.shape[0]

        assert T > 0

        # TODO: adjust to the case when lags > 1
        if T == 1:
            if x0 is not None:
                print("Nothing to sample")
                return
            else:
                return self.observation.sample_x(zs[0])

        if x0 is None:
            x0 = self.observation.sample_x(zs[0])
        else:
            assert x0.shape == (self.D, )
            x0 = check_and_convert_to_tensor(x0, dtype=torch.float64)

        xs = [x0]
        for t in np.arange(1, T):
            x_t = self.observation.sample_x(zs[t], xs[t-1][None, ], return_np=False)
            xs.append(x_t)

        xs = torch.stack(xs)
        assert xs.shape == (T, self.D)
        if return_np:
            return xs.numpy()
        return xs







