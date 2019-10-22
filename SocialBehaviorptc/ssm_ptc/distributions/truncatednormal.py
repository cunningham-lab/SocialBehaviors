import torch
from torch.distributions.distribution import Distribution
from torch.distributions import Normal

import numpy as np

import math
from ssm_ptc.distributions.base_distribution import BaseDistribution
from ssm_ptc.utils import check_and_convert_to_tensor, get_np

from scipy.stats import truncnorm


def normal_log_pdf(kexi):
    out = - 0.5 * math.log(2 * math.pi) - 0.5 * kexi**2
    return out


def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class TruncatedNormal(BaseDistribution):
    """
    Creates a sigmoid-normal distribution parameterized by a mean vector, a covariance matrix and bounds
    Normal -> scaled sigmoid
    """

    def __init__(self, mus, log_sigmas, bounds, device=torch.device('cpu')):
        super(TruncatedNormal, self).__init__()

        self.mus = mus
        self.log_sigmas = log_sigmas

        self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64, device=device)  # mus.shape + (2, )
        # assert self.bounds.shape == self.mus.shape + (2,)

    def sample(self, sample_shape=torch.Size()):
        """

        :param sample_shape: currently consider one
        :return: (D, )
        """

        # first, transform to standard normal
        scale = np.exp(get_np(self.log_sigmas))  # (D, )
        loc = get_np(self.mus)  # (D, )
        bb = (get_np(self.bounds[...,1]) - loc) / scale  # (D, )
        aa = (get_np(self.bounds[...,0]) - loc) / scale  # (D, )
        if sample_shape == ():
            D = bb.shape[0]
            samples = truncnorm.rvs(a=aa, b=bb, loc=loc, scale=scale, size=(D))

            # some adhoc way to fix the infinity in sample
            samples[samples == -np.inf] = get_np(self.bounds[...,1])[samples == -np.inf]
            samples[samples == np.inf] = get_np(self.bounds[...,0])[samples == np.inf]
        else:
            samples = truncnorm.rvs(a=aa, b=bb, loc=loc, scale=scale, size=sample_shape)

        samples = torch.tensor(samples, dtype=torch.float64)
        return samples

    def log_prob(self, data):
        """

        :param data: (..., D)
        :return: (..., D)
        """
        sigma = torch.exp(self.log_sigmas)
        bb = self.bounds[..., 1] - self.mus
        aa = self.bounds[..., 0] - self.mus
        out = normal_log_pdf((data - self.mus) / sigma + 1e-15) - self.log_sigmas - \
              torch.log(normal_cdf(bb / sigma) - normal_cdf(aa / sigma) + 1e-15)
        #out = normal_log_pdf((data-self.mus)/(sigma + 1e-15)+1e-15) - self.log_sigmas - \
         #     torch.log(normal_cdf(bb/(sigma+1e-15)) - normal_cdf(aa/(sigma + 1e-15)) + 1e-15)
        return out

    def pdf(self, data):
        return torch.exp(self.log_prob(data))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    """
    mu = torch.tensor(-8, dtype=torch.float64)
    sigma = torch.tensor(2, dtype=torch.float64)
    log_sigma = torch.log(sigma)
    bounds = torch.tensor([-10, 10], dtype=torch.float64)

    dist = TruncatedNormal(mu, log_sigma, bounds)

    x = np.linspace(-10, 10, 1000)
    data = torch.tensor(x)

    data_pdf = dist.pdf(data)

    plt.plot(x, data_pdf.numpy())
    plt.show()

    from scipy.stats import truncnorm

    a = (-10 + 8) / 2
    b = (10 + 8) / 2
    x = (-10 + 8) / 2
    out = truncnorm.pdf(x=-10, a=a, b=b, loc=-8, scale=2)
    # out = truncnorm.pdf(x=x, a=a, b=b) / 2
    print(out)
    

    # test multi-dimensional
    T = 10
    K = 3
    D = 2
    mus = torch.zeros((T, K, D), dtype=torch.float64)
    log_sigmas = torch.zeros((K, D), dtype=torch.float64)
    bounds = torch.tensor([[0,4], [0, 6]], dtype=torch.float64)

    dist = TruncatedNormal(mus, log_sigmas, bounds)

    data = torch.tensor([[0,0], [0, 1], [0,2], [1,2], [1,3],[2,2], [2,3], [3,2], [3,3], [3,4]], dtype=torch.float64)

    assert data.shape == (T, D)

    log_prob = dist.log_prob(data[:,None])
    
    """
    # test sample
    D = 2
    mus = torch.zeros((D,), dtype=torch.float64) - 8
    log_sigmas = torch.log(2 * torch.ones((D,), dtype=torch.float64))
    bounds = torch.tensor([[-10, 10], [-5, 5]], dtype=torch.float64)

    p = TruncatedNormal(mus=mus, log_sigmas=log_sigmas, bounds=bounds)

    samples = p.sample((5000, 2))

    p1 = TruncatedNormal(mus=mus[0], log_sigmas=log_sigmas[0], bounds=bounds[0])

    x1 = np.linspace(-10, 10, 100)
    data1 = torch.tensor(x1, dtype=torch.float64)

    data1_pdf = p1.pdf(data1)

    plt.hist(samples[:, 0].numpy(), bins=100, density=True);
    plt.plot(data1.numpy(), data1_pdf.numpy())

    p2 = TruncatedNormal(mus=mus[1], log_sigmas=log_sigmas[1], bounds=bounds[1])

    x2 = np.linspace(-5, 5, 100)
    data2 = torch.tensor(x2, dtype=torch.float64)

    data2_pdf = p2.pdf(data2)

    plt.hist(samples[:, 1].numpy(), bins=100, density=True);
    plt.plot(data2.numpy(), data2_pdf.numpy())





