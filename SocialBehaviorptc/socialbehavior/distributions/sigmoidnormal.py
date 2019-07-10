import torch
from torch.distributions.distribution import Distribution
from torch.distributions import Normal

import numpy as np

from socialbehavior.distributions.base_distribution import BaseDistribution
from socialbehavior.utils import check_and_convert_to_tensor


# TODO: subclassing torch Distribution

def log_d_sigmoid(x):
    return torch.log(torch.sigmoid(x)) + torch.log(torch.tensor(1, dtype=x.dtype) - torch.sigmoid(x))


class SigmoidNormal(BaseDistribution):
    r"""
    Creates a sigmoid-normal distribution parameterized by a mean vector, a covariance matrix, bounds and centers
    Normal -> scaled sigmoid
    """

    def __init__(self, mus, log_sigmas, bounds, centers=None, alpha=1.0):
        super(SigmoidNormal, self).__init__()

        self.mus = mus
        self.log_sigmas = log_sigmas

        self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64)  # mus.shape + (2, )
        assert self.bounds.shape == self.mus.shape + (2,)

        if centers is None:
            # set centers to be the centers of the bounds
            self.centers = torch.mean(self.bounds, dim=-1)  # mus.shape
        else:
            self.centers = check_and_convert_to_tensor(centers, dtype=torch.float64)
            assert self.centers.shape == self.mus.shape

        self.alpha = torch.tensor(alpha, dtype=torch.float64)  # smoothness parameter for the scale function

    @property
    def normal_dist(self):
        return Normal(self.mus, torch.exp(self.log_sigmas))

    @staticmethod
    def _inverse_sigmoid(x):
        # x = sigmoid(z) <=> z = sigmoid_inv(x)
        # sigmoid_inv (x) = log sigmoid (x) - log (1 - sigmoid(x))
        return torch.log(x) - torch.log(torch.tensor(1) - x)

    def _scale_fn(self, z):
        """
        f(z) = (upper_bounds - lower_bound) * sigmoid( alpha * (z - offset)) + lowerbound
        :param z: batch_shape + mus.shape
        :return: z_tran: batch_shape + mus.shape
        """

        z_tran = (self.bounds[..., 1] - self.bounds[..., 0]) * torch.sigmoid(self.alpha * (z - self.centers))\
                    + self.bounds[...,0]

        assert z_tran.shape == z.shape

        return z_tran

    def _inverse_scale_fn(self, z_tran):
        """
        f_inv (z_tran) = 1 / alpha * sigmoid_inv ( (z_tran - lower_bound) / (upper_bound - lower_bound) ) + offset
        :param z_tran: (..., D)
        :return: z: (..., D)
        """

        # inverse scale
        x = (z_tran - self.bounds[..., 0]) / (self.bounds[..., 1] - self.bounds[..., 0])
        assert x.shape == z_tran.shape

        # inverse sigmoid
        z = self._inverse_sigmoid(x) / self.alpha + self.centers
        assert z.shape == z_tran.shape

        return z

    def _log_d_scale_fn(self, z):
        """
        element-wise log derivative: log df(z) / dz = log bound_gaps + log sigmoid(z) + log (1-sigmoid(z))
        :param z: (..., D)
        :return:
        """
        out = torch.log((self.bounds[..., 1] - self.bounds[..., 0])) + torch.log(self.alpha) \
              + log_d_sigmoid(self.alpha * (z - self.centers))
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

        out = log_p_z - self._log_d_scale_fn(z)

        return out

    def pdf(self, data):
        return torch.exp(self.log_prob(data))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # D = 1
    bounds = np.array([[1, 3]])
    centers = np.array([2])

    mus = torch.zeros(1, dtype=torch.float64)
    log_sigmas = torch.zeros(1, dtype=torch.float64)
    sn_dist = SigmoidNormal(mus=mus, log_sigmas=log_sigmas, bounds=bounds, centers=centers)

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

    # check log prob and pdf
    # D = 1
    bounds = np.array([[0, 1]])
    centers = np.array([0]) # it should be mus

    mus = torch.zeros(1, dtype=torch.float64)
    log_sigmas = torch.zeros(1, dtype=torch.float64)
    sn_dist = SigmoidNormal(mus=mus, log_sigmas=log_sigmas, bounds=bounds, centers=centers)

    T = 1000
    data = torch.tensor(np.linspace(0, 1, T)[:, None])
    plt.plot(data.numpy(), sn_dist.pdf(data).numpy())
    plt.title("density curve")
    plt.show()

    # check sample
    T = 5000
    D = 1

    bounds = np.array([[1,3]])
    mus = torch.tensor([0], dtype=torch.float64)
    log_sigmas = torch.tensor([0], dtype=torch.float64)
    sn_dist = SigmoidNormal(mus, log_sigmas, bounds)
    samples = sn_dist.sample((T,))

    plt.hist(samples.numpy(), bins=100, density=True)
    plt.plot(samples.numpy(), sn_dist.pdf(samples).numpy(), 'o', label='true pdf')
    plt.legend()
    plt.title("histogram of sampled data")

    plt.show()


