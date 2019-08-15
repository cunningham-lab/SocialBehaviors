import torch
from torch.distributions import MultivariateNormal
from ssm_ptc.transformations import BaseTransformation
from ssm_ptc.transformations.linear import LinearTransformation


class GaussianObservation():
    """
    A mixture of gaussians
    """

    def __init__(self, K, D, M, transformation, mus_init=None, sigmas=None):
        super(GaussianObservation, self).__init__(K, D, M)

        if mus_init is None:
            self.mus_init = torch.eye(self.K, self.D, dtype=torch.float64, requires_grad=True)
        else:
            self.mu_init = torch.tensor(mus_init, dtype=torch.float64, requires_grad=True)

        if sigmas is None:
            self.sigmas = 0.1 * torch.ones(self.K, self.D, self.D, dtype=torch.float64, requires_grad=True)
        else:
            assert sigmas.shape == (self.K, self.D, self.D)
            self.sigmas = torch.tensor(sigmas, dtype=torch.float64, requires_grad=True)

        if isinstance(transformation, str):
            if transformation == 'linear':
                self.transformation = LinearTransformation(K=self.K, d_in=self.D, D=self.D)
        else:
            assert isinstance(self.transformation, BaseTransformation)
            self.transformation = transformation

    @property
    def params(self):
        return [self.mus_init, self.sigmas] + self.transformation.params

    def _compute_mus(self, data):

        T, D = data.shape
        assert D == self.D

        mus = torch.empty(T, self.K, self.D)

        mus[0] = self.mus_init

        mus[1:] = self.transformation.transform(data[1:])

        assert mus.shape == (T, self.K, self.D)
        return mus

    def log_prob(self, data):

        mus = self._compute_mus(data)

        p = MultivariateNormal(mus, self.sigmas)

        out = p.log_prob(data)

        return out

    def sample_x(self, z, xhist):
        """

        :param z: shape ()
        :param xhist: shape (T_pre, D)
        :return: x: shape (D,)
        """

        # no previous x
        if xhist.shape[0] == 0:
            mu = self.mus_init[z]
            return mu + self.sigmas









