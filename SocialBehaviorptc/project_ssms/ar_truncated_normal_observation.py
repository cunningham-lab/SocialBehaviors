import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.observations.base_observation import BaseObservation
from ssm_ptc.distributions.truncatednormal import TruncatedNormal
from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.coupled_transformations.grid_transformation import GridTransformation
from project_ssms.coupled_transformations.lineargrid_transformation import LinearGridTransformation
from project_ssms.coupled_transformations.weightedgrid_transformation import WeightedGridTransformation


TRANSFORMATION_CLASSES = dict(
            grid=GridTransformation,
            lineargrid=LinearGridTransformation,
            weightedgrid=WeightedGridTransformation
            )


class ARTruncatedNormalObservation(BaseObservation):
    def __init__(self, K, D, M=0, lags=1, bounds=None, transformation="grid", transformation_kwargs=None, train_sigma=True):

        super(ARTruncatedNormalObservation, self).__init__(K, D, M)

        self.lags = lags

        if bounds is None:
            raise ValueError("Must provide bounds.")
        self.bounds = check_and_convert_to_tensor(bounds)
        assert self.bounds.shape == (self.D, 2)

        self.mus_init = torch.eye(self.K, self.D, dtype=torch.float64)
        self.log_sigmas_init = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64)
        self.log_sigmas = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, requires_grad=train_sigma)

        if isinstance(transformation, BaseTransformation):
            self.transformation = transformation
        elif isinstance(transformation, str):
            if transformation not in TRANSFORMATION_CLASSES:
                raise Exception("Invalid transformation model: {}. Must be one of {}".
                                format(transformation, list(TRANSFORMATION_CLASSES.keys())))

            transformation_kwargs = transformation_kwargs or {}
            self.transformation = TRANSFORMATION_CLASSES[transformation](K=self.K, D=self.D, M=self.M, lags=self.lags,
                                                                         **transformation_kwargs)

    @property
    def params(self):
        return (self.log_sigmas, ) + self.transformation.params

    @params.setter
    def params(self, values):
        self.log_sigmas = set_param(self.log_sigmas, values[0])
        self.transformation.params = values[1:]

    def permute(self, perm):
        self.mus_init = self.mus_init[perm]
        self.log_sigmas_init = self.log_sigmas_init[perm]

        self.log_sigmas = torch.tensor(self.log_sigmas[perm], requires_grad=self.log_sigmas.requires_grad)
        self.transformation.permute(perm)

    def log_prior(self):
        return self.transformation.log_prior()

    def log_prob(self, data, **memory_args):
        mus = self._compute_mus_for(data, **memory_args)  # (T, K, D)

        dist = TruncatedNormal(mus=mus, log_sigmas=self.log_sigmas, bounds=self.bounds)

        out = dist.log_prob(data[:, None])  # (T, K, D)
        out = torch.sum(out, dim=-1)  # (T, K)
        return out

    def sample_x(self, z, xhist=None, transformation=False, return_np=True, **memory_kwargs):
        """

        :param z: an integer
        :param xhist: (T_pre, D)
        :param transformation: return transformed value as sample value, instead of sampling
        :param return_np: boolean, whether return np.ndarray or torch.tensor
        :return: one sample (D, )
        """
        if xhist is None or xhist.shape[0] == 0:

            mu = self.mus_init[z]  # (D,)
            log_sigma = self.log_sigmas_init[z]
        else:
            # sample from the autoregressive distribution

            T_pre = xhist.shape[0]
            if T_pre < self.lags:
                mu = self.transformation.transform_condition_on_z(z, xhist, **memory_kwargs)  # (D, )
            else:
                mu = self.transformation.transform_condition_on_z(z, xhist[-self.lags:], **memory_kwargs)  # (D, )
            assert mu.shape == (self.D,)

            log_sigma = self.log_sigmas[z]

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
            # TODO: fix domain error here, add debug infor to print arg check
            dist = TruncatedNormal(mus=mu, log_sigmas=log_sigma, bounds=self.bounds)
            samples = dist.sample()

        if return_np:
            return samples.numpy()
        return samples

    def _compute_mus_for(self, data, **memory_args):
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
            mus_rest = self.transformation.transform(data[:-1], **memory_args)
            assert mus_rest.shape == (T - 1, self.K, self.D)

            mus = torch.cat((self.mus_init[None,], mus_rest), dim=0)

        assert mus.shape == (T, self.K, self.D)
        return mus

    def rsample_x(self, z, xhist, expectation=False, **kwargs):
        raise NotImplementedError



