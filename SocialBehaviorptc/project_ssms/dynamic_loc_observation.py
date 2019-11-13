import torch
import numpy as np

from ssm_ptc.observations.base_observation import BaseObservation
from ssm_ptc.distributions.truncatednormal import TruncatedNormal
from ssm_ptc.utils import check_and_convert_to_tensor, set_param, get_np

from project_ssms.coupled_transformations.dynamic_loc_transformation import DynamicLocationTransformation
from project_ssms.single_transformations.single_dynamic_location_transformation import SingleDynamicLocationTransformation


class DynamicLocationObservation(BaseObservation):
    def __init__(self, K, D, M=0, bounds=None):
        assert D == 2 or D == 4, "D muse be either 2 or 4."
        super(DynamicLocationObservation, self).__init__(K, D, M)

        if bounds is None:
            raise ValueError("Must provide bounds.")
        self.bounds = check_and_convert_to_tensor(bounds)
        assert self.bounds.shape == (self.D, 2)

        self.mus_init = torch.eye(self.K, self.D, dtype=torch.float64, requires_grad=True)

        if self.D == 2:
            self.transformation = SingleDynamicLocationTransformation(K=self.K, D=self.D, bounds=get_np(self.bounds))
        else:
            self.transformation = DynamicLocationTransformation(K=self.K, D=self.D)

    @property
    def params(self):
        return (self.mus_init, ) + self.transformation.params

    @params.setter
    def params(self, values):
        self.mus_init = set_param(self.mus_init, values[0])
        self.transformation.params = values[1:]

    @property
    def log_sigmas(self):
        return self.transformation.log_sigmas

    def permute(self, perm):
        self.transformation.permute(perm)

    def log_prob(self, data, **kwargs):
        mus = self._compute_mus_for(data)  # (T, K, D)

        dist = TruncatedNormal(mus=mus, log_sigmas=self.log_sigmas, bounds=self.bounds)

        out = dist.log_prob(data[:, None])  # (T, K, D)
        out = torch.sum(out, dim=-1)  # (T, K)
        return out

    def sample_x(self, z, xhist=None, transformation=False, return_np=True, **memory_kwargs):
        """

        :param z: an integer
        :param xhist: (T_pre, D)
        :param return_np: boolean, whether return np.ndarray or torch.tensor
        :return: one sample (D, )
        """
        if xhist is None or xhist.shape[0] == 0:

            mu = self.mus_init[z]  # (D,)
        else:
            # sample from the autoregressive distribution
            mu = self.transformation.transform_condition_on_z(z, xhist[-1:])  # (D, )
            assert mu.shape == (self.D,)

        if transformation:
            samples = mu
            # some ad-hoc way to address bound issue
            for d in range(self.D):
                if samples[d] <= self.bounds[d,0]:
                    samples[d] = self.bounds[d,0] + 0.1 * torch.rand(1, dtype=torch.float64)
                elif samples[d] >= self.bounds[d,1]:
                    samples[d] = self.bounds[d,1] - 0.1 * torch.rand(1, dtype=torch.float64)

            samples = samples.detach()

        else:
            dist = TruncatedNormal(mus=mu, log_sigmas=self.log_sigmas[z], bounds=self.bounds)
            samples = dist.sample()

        if return_np:
            return samples.numpy()
        return samples

    def _compute_mus_for(self, data):
        """

        :param data: (T, D)
        :param memory_args: for transformation's use
        :return: mus, the mean for data (T, D)
        """
        T, D = data.shape
        assert D == self.D

        if T == 1:
            mus = self.mus_init[None,]
        else:
            mus_rest = self.transformation.transform(data[:-1])
            assert mus_rest.shape == (T - 1, self.K, self.D)

            mus = torch.cat((self.mus_init[None,], mus_rest), dim=0)

        assert mus.shape == (T, self.K, self.D)
        return mus

    def rsample_x(self, z, xhist, transformation=False):
        raise NotImplementedError



