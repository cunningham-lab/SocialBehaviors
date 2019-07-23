"""
the code is mainly based on https://github.com/slinderman/ssm
"""

import torch
import numpy as np
import numpy.random as npr

from ssm_ptc.observations.base_observation import BaseObservations
from ssm_ptc.observations.ar_gaussian_observation import ARGaussianObservation
from ssm_ptc.message_passing.primitives import viterbi
from ssm_ptc.message_passing.normalizer import hmmnorm_cython
from ssm_ptc.utils import check_and_convert_to_tensor, get_np

from tqdm import trange


class HMM:

    def __init__(self, K, D, M=0, observation="gaussian", pi0=None, Pi=None):
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

        # parameter for the softmax distribution
        if pi0 is None:
            self.pi0 = torch.ones(self.K, dtype=torch.float64, requires_grad=True)
        else:
            self.pi0 = check_and_convert_to_tensor(pi0, dtype=torch.float64)
        if Pi is None:
            Pi = 2 * np.eye(K) + .05 * npr.rand(K, K)
            self.Pi = torch.tensor(Pi, dtype=torch.float64, requires_grad=True)
        else:
            self.Pi = check_and_convert_to_tensor(Pi, dtype=torch.float64)

        if isinstance(observation, str):
            if observation == "gaussian":
                self.observation = ARGaussianObservation(self.K, self.D, self.M, transformation='linear')
        elif isinstance(observation, BaseObservations):
            self.observation = observation
        else:
            raise Exception("Invalid observation type.")

    @property
    def init_dist(self):
        return torch.nn.Softmax(dim=0)(self.pi0)

    @property
    def transition_matrix(self):
        return torch.nn.Softmax(dim=1)(self.Pi)

    def sample_z(self, T):
        # sample the time-invariant markov chain only
        z = torch.empty(T, dtype=torch.int)
        pi0 = self.init_dist.detach()
        z[0] = npr.choice(self.K, p=pi0)

        P = self.transition_matrix.detach()  # (K, K)
        for t in range(1, T):
            z[t] = npr.choice(self.K, p=P[z[t - 1]])
        return z

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
        #dummy_data = self.observation.sample_x(0, np.empty(0,), return_np=False)
        #dtype = dummy_data.dtype
        dtype = torch.float64

        if prefix is None:
            # no prefix is given. Sample the initial state as the prefix
            T_pre = 1
            z = torch.empty(T, dtype=torch.int)
            data = torch.empty((T, D), dtype=dtype)

            # sample the first state from the initial distribution
            pi0 = self.init_dist.detach()
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

            # construct the states and data
            z = torch.cat((z_pre, torch.empty(T, dtype=torch.int)))
            assert z.shape == (T_pre + T, )
            data = torch.cat((x_pre, torch.empty((T, D), dtype=dtype)))

        P = self.transition_matrix.detach() # (K, K)
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
        data = check_and_convert_to_tensor(data, torch.float64)

        T = data.shape[0]
        log_pi0 = torch.nn.LogSoftmax(dim=0)(self.pi0)  # (K, )
        log_P = torch.nn.LogSoftmax(dim=1)(self.Pi)  # (K, K)

        if T == 1:
            log_Ps = log_P[None,][:0]
        else:
            log_Ps = log_P[None,].repeat(T-1, 1, 1)  # (T-1, K, K)

        ll = self.observation.log_prob(data)  # (T, K)

        return hmmnorm_cython(log_pi0, log_Ps, ll)

    @property
    def params(self):
        """
        :return: pi0, Pi, mus_init, log_sigmas, As, bs ...
        """
        return [self.pi0, self.Pi] + self.observation.params

    @params.setter
    def params(self, values):
        """only change values, keep requires_grad property"""
        # TODO: test this method
        assert type(values) == list

        self.pi0 = values[0]
        self.Pi = values[1]
        self.observation.params = values[2:]

    @property
    def trainable_params(self):
        """
        :return: the parameters that require grad. maybe helpful for optimization
        """
        out = []
        for param in self.params:
            if param.requires_grad:
                out.append(param)
        return out

    # numpy operation
    def most_likely_states(self, data):
        data = check_and_convert_to_tensor(data)

        log_pi0 = torch.nn.LogSoftmax(dim=0)(self.pi0).detach().numpy()  # (K, )
        log_Ps = torch.nn.LogSoftmax(dim=1)(self.Pi).detach().numpy()  # (K, K)

        log_likes = self.observation.log_prob(data).detach().numpy()
        return viterbi(log_pi0, log_Ps[None,], log_likes)

    def permute(self, perm):
        self.pi0 = self.pi0[perm]
        self.Pi = self.Pi[np.ix_(perm, perm)]
        self.observation.permute(perm)

    # return np
    def sample_condition_on_zs(self, zs, x0=None, return_np=True):
        """
        Given a z sequence, generate samples condition on this sequence.
        :param zs: (T, )
        :param x0: shape (D,)
        :return: generated samples (T, D)
        """

        zs = check_and_convert_to_tensor(zs, dtype=torch.int)
        T = zs.shape[0]

        assert T > 0

        # TODO: test lags
        if T == 1:
            if x0 is not None:
                print("Nothing to sample")
                return
            else:
                return self.observation.sample_x(zs[0])

        if x0 is None:
            x0 = self.observation.sample_x(zs[0], return_np=False)
        else:
            assert x0.shape == (self.D, )

        xs = [x0]
        for t in np.arange(1, T):
            x_t = self.observation.sample_x(zs[t], xs[t-1][None, ], return_np=False)
            xs.append(x_t)

        xs = torch.stack(xs)
        assert xs.shape == (T, self.D)
        if return_np:
            return xs.numpy()
        return xs

    def fit(self, data, optimizer=None, method='adam', num_iters=1000, lr=0.001, lr_scheduler=None):

        pbar = trange(num_iters)

        if optimizer is None:
            if method == 'adam':
                optimizer = torch.optim.Adam(self.params, lr=lr)
            elif method == 'sgd':
                optimizer = torch.optim.SGD(self.params, lr=lr)
            else:
                raise ValueError("method must be chosen from adam and sgd.")

        losses = []
        for i in np.arange(num_iters):

            optimizer.zero_grad()

            loss = self.loss(data)
            loss.backward()
            optimizer.step()

            loss = loss.detach().numpy()
            losses.append(loss)

            if i % 10 == 0:
                pbar.set_description('iter {} loss {:.2f}'.format(i, loss))
                pbar.update(10)

        pbar.close()

        return losses, optimizer








