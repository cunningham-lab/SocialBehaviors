import torch
from torch.distributions.distribution import Distribution
from torch.distributions import Normal

import numpy as np

from ssm_ptc.distributions.base_distribution import BaseDistribution
from ssm_ptc.utils import check_and_convert_to_tensor


# TODO: subclassing torch Distribution

def log_d_sigmoid(x):
    return torch.log(torch.sigmoid(x)) + torch.log(torch.tensor(1, dtype=x.dtype) - torch.sigmoid(x))


class LogitNormal(BaseDistribution):
    r"""
    Creates a sigmoid-normal distribution parameterized by a mean vector, a covariance matrix and bounds
    Normal -> scaled sigmoid
    """

    def __init__(self, mus, log_sigmas, bounds, alpha=1.0):
        super(LogitNormal, self).__init__()

        self.mus = mus
        self.log_sigmas = log_sigmas

        self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64)  # mus.shape + (2, )
        #assert self.bounds.shape == self.mus.shape + (2,)

        self.alpha = torch.tensor(alpha, dtype=torch.float64)
        assert self.alpha.shape == ()

    @property
    def normal_dist(self):
        return Normal(self.mus, torch.exp(self.log_sigmas))

    @staticmethod
    def _inverse_sigmoid(x):
        # logistic function
        # x = sigmoid(z) <=> z = sigmoid_inv(x)
        # sigmoid_inv (x) = log (x) - log (1 - x)
        return torch.log(x) - torch.log(torch.tensor(1) - x)

    def _scale_fn(self, z):
        """
        f(z) = (upper_bounds - lower_bound) * sigmoid(z) + lowerbound
        :param z: batch_shape + mus.shape
        :return: z_tran: batch_shape + mus.shape
        """

        z_tran = (self.bounds[..., 1] - self.bounds[..., 0]) * torch.sigmoid(self.alpha * z) \
                    + self.bounds[...,0]

        assert z_tran.shape == z.shape

        return z_tran

    def _inverse_scale_fn(self, z_tran):
        """
        f_inv (z_tran) = sigmoid_inv ( (z_tran - lower_bound) / (upper_bound - lower_bound) )
        :param z_tran: (..., D)
        :return: z: (..., D)
        """

        # inverse scale
        x = (z_tran - self.bounds[..., 0]) / (self.bounds[..., 1] - self.bounds[..., 0])
        assert x.shape == z_tran.shape

        # inverse sigmoid
        z = self._inverse_sigmoid(x) / self.alpha
        assert z.shape == z_tran.shape

        return z

    def _log_d_scale_fn(self, z):
        """
        element-wise log derivative: log df(z) / dz = log bound_gaps + log sigmoid(z) + log (1-sigmoid(z))
        :param z: (..., D)
        :return:
        """
        out = torch.log((self.bounds[..., 1] - self.bounds[..., 0])) + torch.log(self.alpha) \
              + log_d_sigmoid(self.alpha * z)
        return out

    def rsample(self, sample_shape=torch.Size()):
        # sample from normal
        p = self.normal_dist
        z = p.rsample(sample_shape)
        assert z.shape == sample_shape + self.mus.shape

        # scaled sigmoid transformation
        z_tran = self._scale_fn(z)
        assert z_tran.shape == sample_shape + self.mus.shape
        return z_tran

    def log_prob(self, data):
        """

        :param data: (..., D)
        :return: (..., D)
        """

        z = self._inverse_scale_fn(data)

        p = self.normal_dist
        log_p_z = p.log_prob(z)

        #out1 = log_p_z - self._log_d_scale_fn(z)

        scale_data = (data - self.bounds[..., 0]) / (self.bounds[..., 1] - self.bounds[..., 0])

        out = log_p_z - torch.log((self.bounds[..., 1] - self.bounds[..., 0])) - torch.log(self.alpha) \
                - torch.log(scale_data) - torch.log(torch.tensor(1, dtype=scale_data.dtype) - scale_data)

        #assert torch.allclose(out1, out)

        return out

    def pdf(self, data):
        return torch.exp(self.log_prob(data))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    """
    # D = 1
    bounds = np.array([[1, 3]])

    mus = torch.zeros(1, dtype=torch.float64)
    log_sigmas = torch.zeros(1, dtype=torch.float64)
    sn_dist = SigmoidNormal(mus=mus, log_sigmas=log_sigmas, bounds=bounds)

    # check sigmoid inversion
    z = torch.tensor(np.linspace(-5, 5, 20), dtype=torch.float64)  # (20,)
    x = torch.sigmoid(z)
    z_recovered = sn_dist._inverse_sigmoid(x)

    assert torch.allclose(z, z_recovered)

    # check scale_fn inverse
    y = sn_dist._scale_fn(x)
    x_recovered = sn_dist._inverse_scale_fn(y)
    assert torch.allclose(x, x_recovered)

    # check sample shape
    mus = torch.zeros(2, 3, dtype=torch.float64)
    log_sigmas = torch.zeros(2, 3, dtype=torch.float64)
    bounds = np.tile(np.array([[1,5], [2, 6], [3,8]])[None,], (2,1,1))
    sn_dist = SigmoidNormal(mus=mus, log_sigmas=log_sigmas, bounds=bounds)

    sample_shape = (10, 5)
    samples = sn_dist.sample(sample_shape)
    assert samples.shape == sample_shape + sn_dist.mus.shape
    """

    # check log prob and pdf
    # D = 1
    bounds = np.array([[0, 1]])

    mus = torch.zeros(1, dtype=torch.float64)
    log_sigmas = torch.zeros(1, dtype=torch.float64)
    sn_dist = LogitNormal(mus=mus, log_sigmas=log_sigmas, bounds=bounds)

    T = 100
    data = torch.tensor(np.linspace(0.5, 0.9, T)[:, None])
    plt.plot(data.numpy(), sn_dist.pdf(data).numpy())
    plt.title("density curve")
    plt.show()

    """
    # check sample
    T = 5000
    D = 1

    bounds = np.array([[1,3]])
    mus = torch.tensor([2], dtype=torch.float64)
    log_sigmas = torch.tensor([0], dtype=torch.float64)
    sn_dist = SigmoidNormal(mus, log_sigmas, bounds)
    samples = sn_dist.sample((T,))

    plt.hist(samples.numpy(), bins=100, density=True)
    plt.plot(samples.numpy(), sn_dist.pdf(samples).numpy(), 'o', label='true pdf')
    plt.legend()
    plt.title("histogram of sampled data")

    plt.show()
    
    """


