from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.observations.base_observation import BaseObservations
from ssm_ptc.observations.ar_truncated_normal_observation import ARTruncatedNormalObservation
from ssm_ptc.distributions.truncatednormal import TruncatedNormal
from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.momentum_utils import get_momentum_in_batch, get_momentum

import torch
import numpy as np


def normalize(f, norm=1):
    # normalize across last dimension
    if norm==1:
        f = f / torch.sum(f, dim=-1, keepdim=True)
    elif norm == 2:
        f = f / torch.norm(f, dim=-1, keepdim=True)
    return f


class CoupledMomemtumTransformation(BaseTransformation):
    """
    transformation:
    x^a_t \sim x^a_{t-1} + acc_factor * sigmoid(\alpha_a) \frac{m_t}{lags} + v * sigmoid(Wf(x^a_t-1, x^b_t-1)+b)
    x^b_t \sim x^b_{t-1} + v * sigmoid(\alpha_b) \frac{m_t}{lags} + v * sigmoid(Wf(x^b_t-1, x^a_t-1)+b)


    feature computation
    training mode: receive precomputed feature input
    sampling mode: compute the feature based on previous observation
    """
    def __init__(self, K, D=4, Df=0, lags=2, feature_funcs=None, max_v=np.array([6, 6, 6, 6]), acc_factor=2):
        super(CoupledMomemtumTransformation, self).__init__(K, D)
        assert D == 4
        self.Df = Df

        self.lags = lags
        self.feature_funcs = feature_funcs

        self.max_v = check_and_convert_to_tensor(max_v)
        assert max_v.shape == (self.D, )

        self.acc_factor = acc_factor

        self.alpha = torch.rand(self.K, self.D, dtype=torch.float64, requires_grad=True)

        self.Ws = torch.rand(self.K, self.D, self.Df, dtype=torch.float64, requires_grad=True)
        self.bs = torch.zeros(self.K, self.D, dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return self.alpha, self.Ws, self.bs

    @params.setter
    def params(self, values):
        self.alpha = set_param(self.alpha, values[0])
        self.Ws = set_param(self.Ws, values[1])
        self.bs = set_param(self.bs, values[2])

    def permute(self, perm):
        self.alpha = self.alpha[perm]
        self.Ws = self.Ws[perm]
        self.bs = self.bs[perm]

    def _compute_momentum_vecs(self, inputs):
        # compute normalized momentum vec
        momentum_vec_a = get_momentum_in_batch(inputs[:, 0:2], lags=self.lags)  # (T, 2)
        momentum_vec_b = get_momentum_in_batch(inputs[:, 2:4], lags=self.lags)  # (T, 2)
        momentum_vec = torch.cat([momentum_vec_a, momentum_vec_b], dim=1)  # (T, 4)
        return momentum_vec

    def _compute_features(self, inputs):

        # compute features
        features_a = self.feature_funcs(inputs[..., :2], inputs[..., 2:])
        features_b = self.feature_funcs(inputs[..., 2:], inputs[..., :2])

        return features_a, features_b

    def transform(self, inputs, momentum_vecs=None, features=None):
        """
        Perform the following transformation:
            x^a_t \sim x^a_{t-1} + sigmoid(\alpha_a) \frac{m_t}{lags} + sigmoid(Wf(x^a_t-1, x^b_t-1)+b)
            x^b_t \sim x^b_{t-1} + sigmoid(\alpha_b) \frac{m_t}{lags} + sigmoid(Wf(x^b_t-1, x^a_t-1)+b)

        :param inputs: (T, 4)
        :param momentum_vec: （T, 4), has been normalized by lags
        :param features: (features_a, features_b)
        :return: outputs: (T, K, 4)
        """

        T = inputs.shape[0]

        if momentum_vecs is None:
           momentum_vecs = self._compute_momentum_vecs(inputs)
        assert momentum_vecs.shape == (T, 4)

        if features is None:
            features_a, features_b = self._compute_features(inputs)
        else:
            features_a, features_b = features

        assert features_a.shape == (T, self.Df)
        assert features_b.shape == (T, self.Df)

        # want: (K, 2, Df) x (T, Df) --> (T, K, 2)
        # actual: (K, 2, Df) x (T, 1, Df, 1) -- (T, K, 2, 1)
        features_transformed_a = torch.matmul(self.Ws[:, :2], features_a[:, None, :, None])
        features_transformed_b = torch.matmul(self.Ws[:, 2:], features_b[:, None, :, None])

        features_transformed = torch.cat((features_transformed_a, features_transformed_b), dim=2)
        #  (T, K, 4, 1)
        features_transformed = torch.squeeze(features_transformed, dim=-1) + self.bs

        assert features_transformed.shape == (T, self.K, self.D)

        out1 = torch.sigmoid(self.alpha) * momentum_vecs[:, None, ]  # (T, K, D)
        assert out1.shape == (T, self.K, self.D)

        # instead of normalize, maybe applying a sigmoid?
        # or treat momentum as features
        out2 = torch.sigmoid(features_transformed)

        out = inputs[:, None, ] + self.acc_factor * out1 + self.max_v * out2

        assert out.shape == (T, self.K, self.D)
        return out

    def transform_condition_on_z(self, z, inputs, **kwargs):
        """
        Perform transformation for given z
        :param z: an integer
        :param inputs: (T_pre, D)
        :return: x: (D, )
        """

        momentum_vec = kwargs.get("momentum_vec", None)
        features = kwargs.get("features", None)

        if momentum_vec is None:

            momentum_vec_a = get_momentum(inputs[:, 0:2], lags=self.lags)  # (2, )
            momentum_vec_b = get_momentum(inputs[:, 2:4], lags=self.lags)  # (2, )
            momentum_vec = torch.cat([momentum_vec_a, momentum_vec_b], dim=0)  # (4, )
        assert momentum_vec.shape == (4, )

        if features is None:

            features_a, features_b = self._compute_features(inputs[-1:])
            assert features_a.shape == (1, self.Df)
            assert features_b.shape == (1, self.Df)
            features_a = torch.squeeze(features_a, dim=0)
            features_b = torch.squeeze(features_b, dim=0)
        else:
            features_a, features_b = features

        # (2, Df) * (Df, 1)  -> (2, 1)
        features_transformed_a = torch.matmul(self.Ws[z, :2], features_a[:, None])
        features_transformed_b = torch.matmul(self.Ws[z, 2:], features_b[:, None])
        assert features_transformed_a.shape == (2, 1)
        assert features_transformed_b.shape == (2, 1)

        features_transformed = torch.cat((features_transformed_a, features_transformed_b), dim=0)
        features_transformed = torch.squeeze(features_transformed, dim=-1) + self.bs[z]
        assert features_transformed.shape == (4, )

        # (D, ) * (D,) --> (D, )
        out1 = torch.sigmoid(self.alpha[z]) * momentum_vec
        # (D,)
        out2 = torch.sigmoid(features_transformed)

        out = inputs[-1] + self.acc_factor * out1 + self.max_v * out2
        assert out.shape == (4, )
        return out


#  TODO: set training mode and test (sampling) mode
class CoupledMomentumObservation(BaseObservations):
    """
    Consider a coupled momentum model:

    transformation:
    x^a_t \sim x^a_{t-1} + v * sigmoid(\alpha_a) \frac{m_t}{lags} + v * sigmoid(Wf(x^a_t-1, x^b_t-1)+b)
    x^b_t \sim x^b_{t-1} + v * sigmoid(\alpha_b) \frac{m_t}{lags} + v * sigmoid(Wf(x^b_t-1, x^a_t-1)+b)

    constrained observation:
    ar_truncated_normal_observation

    feature computation
    training mode: receive precomputed feature input
    sampling mode: compute the feature based on previous observation

    """
    def __init__(self, K, D, M=0, lags=50, Df=1, feature_func=None, mus_init=None, sigmas=None,
                 bounds=None, max_v=np.array([6, 6, 6, 6]), acc_factor=2, train_sigma=True):
        super(CoupledMomentumObservation, self).__init__(K, D, M)

        assert lags > 1

        self.lags = lags
        self.transformation = CoupledMomemtumTransformation(K, D, Df, lags=self.lags,
                                                            feature_funcs=feature_func,
                                                            max_v=max_v, acc_factor=acc_factor)

        # consider diagonal covariance
        if sigmas is None:
            self.log_sigmas = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64, requires_grad=train_sigma)
        else:
            # TODO: assert sigmas positive
            assert sigmas.shape == (self.K, self.D)
            self.log_sigmas = torch.tensor(np.log(sigmas), dtype=torch.float64, requires_grad=train_sigma)

        if bounds is None:
            raise ValueError("Please provide bounds.")
            # default bound for each dimension is [0,1]
            #self.bounds = torch.cat((torch.zeros(self.D, dtype=torch.float64)[:, None],
             #                        torch.ones(self.D, dtype=torch.float64)[:, None]), dim=1)
        else:
            self.bounds = check_and_convert_to_tensor(bounds, dtype=torch.float64)
            assert self.bounds.shape == (self.D, 2)

        if mus_init is None:
            self.mus_init = torch.eye(self.K, self.D, dtype=torch.float64, requires_grad=True)
        else:
            self.mus_init = torch.tensor(mus_init, dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return (self.mus_init, self.log_sigmas) + self.transformation.params

    @params.setter
    def params(self, values):
        self.mus_init = set_param(self.mus_init, values[0])
        self.log_sigmas = set_param(self.log_sigmas, values[1])
        self.transformation.params = values[2:]

    def _compute_mus_for(self, data, momentum_vecs=None, features=None):
        """
        compute the mean vector for each observation (using the previous observation, or mus_init)
        :param data: (T,D)
        :return: mus: (T, K, D)
        """
        T, D = data.shape
        assert D == self.D

        if T == 1:
            mus = self.mus_init[None, ]
        else:
            mus_rest = self.transformation.transform(data[:-1], momentum_vecs=momentum_vecs, features=features)
            assert mus_rest.shape == (T-1, self.K, self.D)

            mus = torch.cat((self.mus_init[None,], mus_rest), dim=0)

        assert mus.shape == (T, self.K, self.D)
        return mus

    def log_prob(self, data, momentum_vecs=None, features=None):
        mus = self._compute_mus_for(data, momentum_vecs=momentum_vecs, features=features)  # (T, K, D)

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
            if T_pre < self.lags:
                mu = self.transformation.transform_condition_on_z(z, xhist, **kwargs)  # (D, )
            else:
                mu = self.transformation.transform_condition_on_z(z, xhist[-self.lags:], **kwargs)  # (D, )
            assert mu.shape == (self.D,)

        dist = TruncatedNormal(mus=mu, log_sigmas=self.log_sigmas[z], bounds=self.bounds)

        samples = dist.sample()
        if return_np:
            return samples.numpy()
        return samples

    def rsample_x(self, z, xhist):
        raise NotImplementedError


