import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from ssm_ptc.transitions.stationary_transition import StationaryTransition
from ssm_ptc.utils import set_param, ensure_args_are_lists_of_tensors


def dirichlet_logpdf(ps, alpha):
    """
    torch version
    :param ps: data, a vector of length K
    :param alpha: concentration parameters of the dirichlet distribution, a vector of length K
    :return: log pdf, a scalar
    """
    assert torch.all(alpha > 0), "The concentration parameters alpha should be all positive."
    assert torch.allclose(torch.sum(ps), torch.tensor(1, dtype=ps.dtype)), "The summation of probabilities should be 1."
    return torch.sum(torch.log(ps) * (alpha - 1.0), dim=-1) + torch.lgamma(torch.sum(alpha, dim=-1)) -\
           torch.sum(torch.lgamma(alpha), dim=-1)


class StickyTransition(StationaryTransition):
    """
        Upweight the self transition prior.

        pi_k ~ Dir(alpha + kappa * e_k)
    """

    def __init__(self, K, D, M=0, Pi=None, alpha=1, kappa=10):
        super(StickyTransition, self).__init__(K, D, M, Pi)

        # not tensor
        self.alpha = alpha
        self.kappa = kappa

    def log_prior(self):
        K = self.K

        Ps = torch.nn.Softmax(dim=-1)(self.logits)
        lp = 0
        for k in range(K):
            alpha = torch.tensor(self.alpha * np.ones(K) + self.kappa * (np.arange(K) == k), dtype=torch.float64)
            lp += dirichlet_logpdf(Ps[k], alpha)
        return lp


class InputDrivenTransition(StickyTransition):
    """
        Hidden Markov Model whose transition probabilities are
        determined by a generalized linear model applied to the
        exogenous input v_n.

        P(z_t | v_t, z_{t-1}) \sim Softmax(W v_t + P_{z_{t-1}})
    """

    def __init__(self, K, D, M=0, Pi=None, alpha=1, kappa=100, l2_penalty=0.0, use_bias=False):
        super(InputDrivenTransition, self).__init__(K, D, M, Pi, alpha, kappa)

        self.Ws = nn.Parameter(torch.tensor(npr.rand(K, M), dtype=torch.float64), requires_grad=True)

        self.use_bias = use_bias
        self.bs = nn.Parameter(torch.zeros(K, dtype=torch.float64), requires_grad=self.use_bias)

        # penalty for Ws
        self.l2_penalty = l2_penalty

    @property
    def params(self):
        return self.Pi, self.Ws, self.bs

    @params.setter
    def params(self, values):
        self.Pi = set_param(self.Pi, values[0])
        self.Ws = set_param(self.Ws, values[1])

    def permute(self, perm):
        self.Pi = nn.Parameter(torch.tensor(self.Pi[np.ix_(perm, perm)]), requires_grad=True)
        self.Ws = nn.Parameter(torch.tensor(self.Ws[perm]), requires_grad=True)
        self.bs = nn.Parameter(torch.tensor(self.bs[perm]), requires_grad=self.use_bias)

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

    @ensure_args_are_lists_of_tensors
    def initialize(self, datas, inputs):
        raise NotImplementedError








