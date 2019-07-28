import torch
import numpy as np
import numpy.random as npr

from ssm_ptc.transitions.base_transition import BaseTransition
from ssm_ptc.utils import set_param


def dirichlet_logpdf(ps, alpha):
    # TODO: test this method
    return ((torch.log(ps) * torch.sum(alpha - 1.0, dim=-1)) +
                torch.lgamma(torch.sum(alpha, dim=-1)) -
                torch.sum(torch.lgamma(alpha), dim=-1))


class StickyTransition(BaseTransition):
    """
        Upweight the self transition prior.

        pi_k ~ Dir(alpha + kappa * e_k)
    """

    def __init__(self, K, D, M=0, Pi=None, alpha=1, kappa=10):
        super(StickyTransition, self).__init__(K, D, M)

        if Pi is None:
            Pi = 2 * np.eye(K) + .05 * npr.rand(K, K)
        else:
            assert isinstance(Pi, np.ndarray)
            assert Pi.shape == (K, K)

        self.Pi = torch.tensor(Pi, dtype=torch.float64, requires_grad=True)

        # not tensor
        self.alpha = alpha
        self.kappa = kappa

    @property
    def params(self):
        return self.Pi,

    @params.setter
    def params(self, values):
        self.Pi = set_param(self.Pi, values[0])

    def log_prior(self):
        K = self.K

        Ps = torch.nn.Softmax(dim=-1)(self.Pi)
        lp = 0
        for k in range(K):
            alpha = torch.tensor(self.alpha * np.ones(K) + self.kappa * (np.arange(K) == k), dtype=torch.float64)
            lp += dirichlet_logpdf(Ps[k], alpha)
        return lp

    def permute(self, perm):
        self.Pi = self.Pi[np.ix_(perm, perm)]

    def transition_matrix(self, data, input, log=False):
        raise NotImplementedError


class InputDrivenTransition(StickyTransition):
    """
        Hidden Markov Model whose transition probabilities are
        determined by a generalized linear model applied to the
        exogenous input v_n.

        P(z_t | v_t, z_{t-1}) \sim Softmax(W v_t + P_{z_{t-1}})
    """

    def __init__(self, K, D, M=0, Pi=None, alpha=1, kappa=10, l2_penalty=0.0, use_bias=False):
        super(InputDrivenTransition, self).__init__(K, D, M, Pi, alpha, kappa)

        self.Ws = torch.tensor(npr.rand(K, M), dtype=torch.float64, requires_grad=True)

        self.bs = torch.zeros(K, dtype=torch.float64, requires_grad=use_bias)

        # penalty for Ws
        self.l2_penalty = l2_penalty

    @property
    def params(self):
        return (self.Pi, self.Ws, self.bs)

    @params.setter
    def params(self, values):
        self.Pi = set_param(self.Pi, values[0])
        self.Ws = set_param(self.Ws, values[1])

    def permute(self, perm):
        self.Pi = self.Pi[np.ix_(perm, perm)]
        self.Ws = self.Ws[perm]

    def log_prior(self):
        lp = super(InputDrivenTransition, self).log_prior()
        lp = lp + np.sum(-0.5 * self.l2_penalty * self.Ws ** 2)
        return lp

    def transition_matrix(self, data, input, log=False):
        """
        # TODO: test this
        :param data: (T, D)
        :param input: (T, D)
        :return:
        """
        T = data.shape[0]
        assert input.shape[0] == T

        # Previous state effect
        Pi = self.Pi[None,].repeat(T-1, 1, 1)
        assert Pi.shape == (T-1, self.K, self.K)

        # Input effect
        input_effect = torch.matmul(input[1:], self.Ws.transpose(0, 1)) # (T-1, K)
        assert input_effect.shape == (T-1, self.K)

        Pi = Pi + input_effect[:, None, :]
        if log:
            out = torch.nn.LogSoftmax(dim=-1)(Pi)
        else:
            out = torch.nn.Softmax(dim=-1)(Pi)
        assert out.shape == (T-1, self.K, self.K)
        return out








