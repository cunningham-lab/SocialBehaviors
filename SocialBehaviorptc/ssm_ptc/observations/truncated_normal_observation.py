import torch
import numpy as np

from ssm_ptc.observations.base_observation import BaseObservation
from ssm_ptc.distributions.truncatednormal import TruncatedNormal
from ssm_ptc.utils import check_and_convert_to_tensor, set_param


class TruncatedNormalObservation(BaseObservation):
    def __init__(self, K, D, M=0, bounds=None, train_sigma=True):

        super(TruncatedNormalObservation, self).__init__(K, D, M)

        if bounds is None:
            raise ValueError("Must provide bounds.")
        self.bounds = check_and_convert_to_tensor(bounds)
        assert self.bounds.shape == (self.D, 2)  # min max

        # random init between bounds
        mus_val = np.random.rand(K, D)
        for d in range(D):
            mus_val[:, d] = (bounds[d, 1] - bounds[d, 0]) * mus_val[:,d] + bounds[d, 0]
        self.mus = torch.tensor(mus_val, dtype=torch.float64, requires_grad=True)
        self.log_sigmas = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, requires_grad=train_sigma)

    @property
    def params(self):
        return self.mus, self.log_sigmas

    @params.setter
    def params(self, values):
        self.mus = set_param(self.mus, values[0])
        self.log_sigmas = set_param(self.log_sigmas, values[1])

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.log_sigmas = torch.tensor(self.log_sigmas[perm], requires_grad=self.log_sigmas.requires_grad)

    def log_prob(self, data, **memory_args):
        T, D = data.shape
        # (1, K, D) * (T, 1, 1) --> (T, K, D)
        mus = self.mus[None, ] * torch.ones((T, 1, 1), dtype=self.mus.dtype)
        assert mus.shape == (T, self.K, self.D)

        dist = TruncatedNormal(mus=mus, log_sigmas=self.log_sigmas, bounds=self.bounds)

        out = dist.log_prob(data[:, None])  # (T, K, D)
        assert out.shape == (T, self.K, D)
        out = torch.sum(out, dim=-1)  # (T, K)
        return out

    def sample_x(self, z, xhist=None, transformation=False, return_np=True, **memory_kwargs):
        """

        :param z: an integer
        :param xhist: (T_pre, D)
        :param return_np: boolean, whether return np.ndarray or torch.tensor
        :return: one sample (D, )
        """

        if transformation:
            samples = self.mus[z]
            # some ad-hoc way to address bound issue
            for d in range(self.D):
                if samples[d] <= self.bounds[d,0]:
                    samples[d] = self.bounds[d,0] + 0.1 * torch.rand(1, dtype=torch.float64)
                elif samples[d] >= self.bounds[d,1]:
                    samples[d] = self.bounds[d,1] - 0.1 * torch.rand(1, dtype=torch.float64)

            samples = samples.detach()

        else:
            dist = TruncatedNormal(mus=self.mus[z], log_sigmas=self.log_sigmas[z], bounds=self.bounds)
            samples = dist.sample()

        if return_np:
            return samples.numpy()
        return samples

    def rsample_x(self, z, xhist, transformation=False):
        raise NotImplementedError



