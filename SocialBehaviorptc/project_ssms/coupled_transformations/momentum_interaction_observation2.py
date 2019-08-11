from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.observations.base_observation import BaseObservation
from ssm_ptc.observations.ar_truncated_normal_observation import ARTruncatedNormalObservation
from ssm_ptc.distributions.truncatednormal import TruncatedNormal
from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.momentum_utils import get_momentum_in_batch, get_momentum

import torch
import numpy as np


def normalize(f, norm=1):
    # normalize across last dimension
    if norm == 1:
        f = f / torch.sum(f, dim=-1, keepdim=True)
    elif norm == 2:
        f = f / torch.norm(f, dim=-1, keepdim=True)
    return f


class MomentumInteractionTransformation(BaseTransformation):
    """
    transformation:
    x^a_t \sim x^a_{t-1} + momentum_factor * sigmoid(\alpha_a) \frac{m_t}{momentum_lags}
                    + interaction_factor * sigmoid(beta_a) o_t^a + + v_max * sigmoid(x W_a +b_a)
    x^a_t \sim x^a_{t-1} + momentum_factor * sigmoid(\alpha_b) \frac{m_t}{momentum_lags}
                    + interaction_factor * sigmoid(beta_b) o_t^b + + v_max * sigmoid(x W_b +b_b)

    constants: momentum_factor=2, interaction_factor=2, momentum_lags=30, v_max=[6,6,6,6]
    variables: alpha, beta, W, b

    feature computation
    training mode: receive precomputed feature input
    sampling mode: compute the feature based on previous observation
    """

    def __init__(self, K, D=4, momentum_lags=30, momentum_weights=None,
                 max_v=np.array([6, 6, 6, 6]), m_factor=2, i_factor=2):
        super(MomentumInteractionTransformation, self).__init__(K, D)
        assert D == 4

        self.momentum_lags = momentum_lags

        # TODO: this wieghts can be learnable (?)
        if momentum_weights is None:
            self.momentum_weights = torch.ones(momentum_lags, dtype=torch.float64)
        else:
            self.momentum_weights = check_and_convert_to_tensor(momentum_weights)

        self.max_v = check_and_convert_to_tensor(max_v)
        assert max_v.shape == (self.D,)

        self.m_factor = m_factor
        self.i_factor = i_factor

        self.alpha = torch.rand(self.K, self.D, dtype=torch.float64, requires_grad=True)
        self.beta = torch.rand(self.K, self.D, dtype=torch.float64, requires_grad=True)

        self.Ws = torch.rand(self.K, self.D, self.D, dtype=torch.float64, requires_grad=True)
        self.bs = torch.zeros(self.K, self.D, dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return self.alpha, self.beta, self.Ws, self.bs

    @params.setter
    def params(self, values):
        self.alpha = set_param(self.alpha, values[0])
        self.beta = set_param(self.beta, values[1])
        self.Ws = set_param(self.Ws, values[2])
        self.bs = set_param(self.bs, values[3])

    def permute(self, perm):
        self.alpha = self.alpha[perm]
        self.beta = self.beta[perm]
        self.Ws = self.Ws[perm]
        self.bs = self.bs[perm]

    @staticmethod
    def _compute_momentum_vecs(inputs, lags, weights=None):
        # compute normalized momentum vec
        if weights is None:
            weights = torch.ones(lags, dtype=torch.float64)
        else:
            weights = check_and_convert_to_tensor(weights)
        momentum_vec_a = get_momentum_in_batch(inputs[:, 0:2], lags=lags, weights=weights)  # (T, 2)
        momentum_vec_b = get_momentum_in_batch(inputs[:, 2:4], lags=lags, weights=weights)  # (T, 2)
        momentum_vec = torch.cat([momentum_vec_a, momentum_vec_b], dim=1)  # (T, 4)
        return momentum_vec

    @staticmethod
    def _compute_direction_vecs(inputs):
        """

        :param data: (T, D) or (D,)
        :return: (T, D) or (D,)
        """
        vecs_a = normalize(inputs[..., 2:] - inputs[..., :2], 2)
        vecs = torch.cat((vecs_a, -vecs_a), dim=-1)
        assert vecs.shape == inputs.shape
        return vecs

    def transform(self, inputs, momentum_vecs=None, interaction_vecs=None):
        """
        Perform the following transformation:
            x^a_t \sim x^a_{t-1} + momentum_factor * sigmoid(\alpha_a) \frac{m_t}{momentum_lags}
                    + interaction_factor * sigmoid(beta) <v_t, o_t>v_t + + v_max * sigmoid(W_a x+b_a)
        x^a_t \sim x^a_{t-1} + momentum_factor * sigmoid(\alpha_a) \frac{m_t}{momentum_lags}
                    + interaction_factor * sigmoid(beta) <v_t, o_t>v_t + + v_max * sigmoid(W_b x+b_b)

        :param inputs: (T, 4)
        :param momentum_vec: （T, 4), has been normalized by momentum_lags
        :param interaction_vecs: (T, 4)
        :return: outputs: (T, K, 4)
        """

        T = inputs.shape[0]

        if momentum_vecs is None:
            momentum_vecs = self._compute_momentum_vecs(inputs, self.momentum_lags, self.momentum_weights)
        assert momentum_vecs.shape == (T, 4)

        if interaction_vecs is None:
            interaction_vecs = self._compute_direction_vecs(inputs)

        assert interaction_vecs.shape == (T, 4)

        out1 = torch.sigmoid(self.alpha) * momentum_vecs[:, None, ]  # (T, K, D)
        assert out1.shape == (T, self.K, self.D)

        out2 = torch.sigmoid(self.beta) * interaction_vecs[:, None, ]  # (T, K, D)

        # Ws: (K, D, D) inputs: (T, D), bs: (K, D)
        # (1, T, D), (K, D, D) *  -> (K, T, D)
        assert inputs[None,].shape == (1, T, self.D)
        out3 = torch.matmul(inputs[None,], self.Ws)
        out3 = out3.transpose(0, 1)
        assert out3.shape == (T, self.K, self.D)
        out3 = torch.sigmoid(out3 + self.bs)

        out = inputs[:, None, ] + self.m_factor * out1 + self.i_factor * out2 + self.max_v * out3

        assert out.shape == (T, self.K, self.D)
        return out

    def transform_condition_on_z(self, z, inputs, **kwargs):
        """
        Perform transformation for given z
        :param z: an integer
        :param inputs: (T_pre, D)
        :param kwargs: supposedly, momentum_vec = (D, ), interaction_vec = (D, )
        :return: x: (D, )
        """

        momentum_vec = kwargs.get("momentum_vec", None)
        interaction_vec = kwargs.get("interaction_vec", None)

        if momentum_vec is None:
            momentum_vec_a = get_momentum(inputs[:, 0:2], lags=self.momentum_lags,
                                          weights=self.momentum_weights)  # (2, )
            momentum_vec_b = get_momentum(inputs[:, 2:4], lags=self.momentum_lags,
                                          weights=self.momentum_weights)  # (2, )
            momentum_vec = torch.cat([momentum_vec_a, momentum_vec_b], dim=0)  # (4, )
        else:
            momentum_vec = check_and_convert_to_tensor(momentum_vec, dtype=torch.float64)
        assert momentum_vec.shape == (4,)

        if interaction_vec is None:
            interaction_vec = self._compute_direction_vecs(inputs[-1])
        else:
            interaction_vec = interaction_vec
        assert interaction_vec.shape == (4,)

        # (D, ) * (D,) --> (D, )
        out1 = torch.sigmoid(self.alpha[z]) * momentum_vec

        out2 = torch.sigmoid(self.beta[z]) * interaction_vec

        #  (1, D) * (D, D) --> (1, D)
        out3 = torch.squeeze(torch.matmul(inputs[-1:], self.Ws[z]), dim=0)  # (D,)
        out3 = torch.sigmoid(out3 + self.bs[z])

        out = inputs[-1] + self.m_factor * out1 + self.i_factor * out2 + self.max_v * out3
        assert out.shape == (4,)
        return out


class MomentumInteractionObservation(BaseObservation):
    """
    Consider a coupled momentum model:

    transformation:
    x^a_t \sim x^a_{t-1} + v * sigmoid(\alpha_a) \frac{m_t}{momentum_lags} + v * sigmoid(Wf(x^a_t-1, x^b_t-1)+b)
    x^b_t \sim x^b_{t-1} + v * sigmoid(\alpha_b) \frac{m_t}{momentum_lags} + v * sigmoid(Wf(x^b_t-1, x^a_t-1)+b)

    constrained observation:
    ar_truncated_normal_observation

    feature computation
    training mode: receive precomputed feature input
    sampling mode: compute the feature based on previous observation

    """

    def __init__(self, K, D, M=0, bounds=None, mus_init=None, sigmas=None, train_sigma=True, **transition_kwargs):
        super(MomentumInteractionObservation, self).__init__(K, D, M)

        self.momentum_lags = transition_kwargs.get("momentum_lags", None)
        if self.momentum_lags is None:
            raise ValueError("Must provide momentum lags.")

        self.transformation = MomentumInteractionTransformation(K, D, **transition_kwargs)

        # consider diagonal covariance
        if sigmas is None:
            self.log_sigmas = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, requires_grad=train_sigma)
        else:
            # TODO: assert sigmas positive
            assert sigmas.shape == (self.K, self.D)
            self.log_sigmas = torch.tensor(np.log(sigmas), dtype=torch.float64, requires_grad=train_sigma)

        if bounds is None:
            raise ValueError("Please provide bounds.")
        else:
            self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64)
            assert self.bounds.shape == (self.D, 2)

        if mus_init is None:
            self.mus_init = torch.eye(self.K, self.D, dtype=torch.float64)
        else:
            self.mus_init = torch.tensor(mus_init, dtype=torch.float64)

        self.log_sigmas_init = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64)

    @property
    def params(self):
        return (self.log_sigmas,) + self.transformation.params

    @params.setter
    def params(self, values):
        self.log_sigmas = set_param(self.log_sigmas, values[0])
        self.transformation.params = values[1:]

    def permute(self, perm):
        self.mus_init = self.mus_init[perm]
        self.log_sigmas_init = self.log_sigmas_init[perm]
        self.log_sigmas = self.log_sigmas[perm]
        self.transformation.permute(perm)

    def _compute_mus_for(self, data, momentum_vecs=None, interaction_vecs=None):
        """
        compute the mean vector for each observation (using the previous observation, or mus_init)
        :param data: (T,D)
        :return: mus: (T, K, D)
        """
        T, D = data.shape
        assert D == self.D

        if T == 1:
            mus = self.mus_init[None,]
        else:
            mus_rest = self.transformation.transform(data[:-1],
                                                     momentum_vecs=momentum_vecs, interaction_vecs=interaction_vecs)
            assert mus_rest.shape == (T - 1, self.K, self.D)

            mus = torch.cat((self.mus_init[None,], mus_rest), dim=0)

        assert mus.shape == (T, self.K, self.D)
        return mus

    def log_prob(self, data, momentum_vecs=None, interaction_vecs=None):
        mus = self._compute_mus_for(data, momentum_vecs=momentum_vecs, interaction_vecs=interaction_vecs)  # (T, K, D)

        dist = TruncatedNormal(mus=mus, log_sigmas=self.log_sigmas, bounds=self.bounds)

        out = dist.log_prob(data[:, None])  # (T, K, D)
        out = torch.sum(out, dim=-1)  # (T, K)
        return out

    def sample_x(self, z, xhist=None, return_np=True, **kwargs):
        """

        :param z: ()
        :param xhist: (T_pre, D)
        :param return_np:
        :return: x, shape (D, )
        """

        # currently only support non-reparameterizable rejection sampling

        # no previous x
        if xhist is None or xhist.shape[0] == 0:
            mu = self.mus_init[z]  # (D,)
        else:
            # sample from the autoregressive distribution

            T_pre = xhist.shape[0]
            if T_pre < self.momentum_lags:
                mu = self.transformation.transform_condition_on_z(z, xhist, **kwargs)  # (D, )
            else:
                mu = self.transformation.transform_condition_on_z(z, xhist[-self.momentum_lags:], **kwargs)  # (D, )
            assert mu.shape == (self.D,)

        dist = TruncatedNormal(mus=mu, log_sigmas=self.log_sigmas[z], bounds=self.bounds)

        samples = dist.sample()
        if return_np:
            return samples.numpy()
        return samples

    def rsample_x(self, z, xhist):
        raise NotImplementedError

