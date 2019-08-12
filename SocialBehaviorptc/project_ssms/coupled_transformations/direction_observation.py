from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.observations.base_observation import BaseObservation
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


class DirectionTransformation(BaseTransformation):
    """
    transformation:
    x^a_t \sim x^a_{t-1} + acc_factor * [ sigmoid(W^a_0) m_t  + \sum_{i=1}^{Df} sigmoid(W^a_i) f_i ]
    x^b_t \sim x^b_{t-1} + acc_factor * [ sigmoid(W^b_0) m_t  + \sum_{i=1}^{Df} sigmoid(W^b_i) f_i ]


    feature computation
    training mode: receive precomputed feature input
    sampling mode: compute the feature based on previous observation
    """
    def __init__(self, K, D=4, Df=None, momentum_lags=2, momentum_weights=None,
                 feature_vec_func=None, acc_factor=2):
        """

        :param K: number of hidden states
        :param D: dimension of observations
        :param Df: number of direction vectors (not including the momentum vectors)
        :param momentum_lags: number of time lags to accumulate the momentum
        :param momentum_weights: weights for weighted linear regression
        :param feature_vec_func: function to compute the featured direction vector --> (T, 2*Df)
        :param acc_factor: acceleration factor, for the purpose of speed control
        """
        super(DirectionTransformation, self).__init__(K, D)
        assert D == 4

        if Df is None:
            raise ValueError("Please provide number of features")
        self.Df = Df

        self.momentum_lags = momentum_lags

        if momentum_weights is None:
            self.momentum_weights = torch.ones(momentum_lags, dtype=torch.float64)
        else:
            self.momentum_weights = check_and_convert_to_tensor(momentum_weights)

        if feature_vec_func is None:
            raise ValueError("Must provide feature funcs.")
        self.feature_vec_func = feature_vec_func

        self.acc_factor = acc_factor  # int

        self.Ws = torch.rand(self.K, 2, 1 + self.Df, dtype=torch.float64, requires_grad=True)

    @property
    def params(self):
        return self.Ws,

    @params.setter
    def params(self, values):
        self.Ws = set_param(self.Ws, values[0])

    def permute(self, perm):
        self.Ws = self.Ws[perm]

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
    def _compute_features(feature_vec_func, inputs):
        """
        :param feature_vec_func:
        :param inputs: (T, 4)
        :return: a tuple, each is of shape (T, 2*Df)
        """
        features_a = feature_vec_func(inputs[..., :2], inputs[..., 2:])
        features_b = feature_vec_func(inputs[..., 2:], inputs[..., :2])

        return features_a, features_b

    def transform(self, inputs, **memory_kwargs):
        """
        Perform the following transformation:
            x^a_t \sim x^a_{t-1} + acc_factor * [ sigmoid(W^a_0) m_t  + \sum_{i=1}^{Df} sigmoid(W^a_i) f_i ]
            x^b_t \sim x^b_{t-1} + acc_factor * [ sigmoid(W^b_0) m_t  + \sum_{i=1}^{Df} sigmoid(W^b_i) f_i ]

        :param inputs: (T, 4)
        :param momentum_vec: （T, 4), has been normalized by momentum_lags
        :param features: (features_a, features_b), each is of shape (T, Df)
        :return: outputs: (T, K, 4)
        """

        T = inputs.shape[0]

        momentum_vecs = memory_kwargs.get("momentum_vecs", None)
        features = memory_kwargs.get("features", None)

        if momentum_vecs is None:
           momentum_vecs = self._compute_momentum_vecs(inputs, self.momentum_lags, self.momentum_weights)
        assert momentum_vecs.shape == (T, 4)

        if features is None:
            features_a, features_b = self._compute_features(self.feature_vec_func, inputs)
        else:
            features_a, features_b = features

        assert features_a.shape == (T, self.Df, 2)
        assert features_b.shape == (T, self.Df, 2)

        all_vecs_a = torch.cat((momentum_vecs[:, None, :2], features_a), dim=1)  # (T, 1+Df, 2)
        all_vecs_b = torch.cat((momentum_vecs[:, None, 2:], features_b), dim=1)  # (T, 1+Df, 2)

        assert all_vecs_a.shape == (T, 1 + self.Df, 2)
        assert all_vecs_b.shape == (T, 1 + self.Df, 2)

        # (K, 1+Df) * (T, 1+Df, 2) -> (T, K, 2)
        out_a = torch.matmul(torch.sigmoid(self.Ws[:,0]), all_vecs_a)
        assert out_a.shape == (T, self.K, 2)

        out_b = torch.matmul(torch.sigmoid(self.Ws[:,1]), all_vecs_b)
        assert out_b.shape == (T, self.K, 2)

        out = torch.cat((out_a, out_b), dim=-1)
        assert out.shape == (T, self.K, self.D)

        out = inputs[:, None, ] + self.acc_factor * out

        assert out.shape == (T, self.K, self.D)
        return out

    def transform_condition_on_z(self, z, inputs, **memory_kwargs):
        """
        Perform transformation for given z
        :param z: an integer
        :param inputs: (T_pre, D)
        :param memory_kwargs: supposedly, momentum_vec = (D, ), features = (features_a, features_b), each of shape (Df, )
        :return: x: (D, )
        """

        momentum_vec = memory_kwargs.get("momentum_vec", None)
        features = memory_kwargs.get("features", None)

        if momentum_vec is None:
            momentum_vec_a = get_momentum(inputs[:, 0:2], lags=self.momentum_lags, weights=self.momentum_weights)  # (2, )
            momentum_vec_b = get_momentum(inputs[:, 2:4], lags=self.momentum_lags, weights=self.momentum_weights)  # (2, )
            momentum_vec = torch.cat([momentum_vec_a, momentum_vec_b], dim=0)  # (4, )
        else:
            momentum_vec = check_and_convert_to_tensor(momentum_vec, dtype=torch.float64)
        assert momentum_vec.shape == (4, )

        if features is None:
            features_a, features_b = self._compute_features(self.feature_vec_func, inputs[-1:])
            assert features_a.shape == (1, self.Df, 2)
            assert features_b.shape == (1, self.Df, 2)
            features_a = torch.squeeze(features_a, dim=0)
            features_b = torch.squeeze(features_b, dim=0)
        else:
            features_a, features_b = features
            features_a = check_and_convert_to_tensor(features_a, dtype=torch.float64)
            features_b = check_and_convert_to_tensor(features_b, dtype=torch.float64)

        assert features_a.shape == (self.Df, 2)
        assert features_b.shape == (self.Df, 2)

        all_vecs_a = torch.cat((momentum_vec[None, :2], features_a), dim=0)  # (1+Df, 2)
        all_vecs_b = torch.cat((momentum_vec[None, 2:], features_b), dim=0)  # (1+Df, 2)

        assert all_vecs_a.shape == (1 + self.Df, 2)
        assert all_vecs_b.shape == (1 + self.Df, 2)

        # (1, 1+Df) * (1+Df, 2) -> (1, 2)
        out_a = torch.matmul(torch.sigmoid(self.Ws[z, 0:1]), all_vecs_a)
        assert out_a.shape == (1, 2)

        out_b = torch.matmul(torch.sigmoid(self.Ws[z, 1:2]), all_vecs_b)
        assert out_b.shape == (1, 2)

        out = torch.cat((out_a, out_b), dim=-1)
        out = torch.squeeze(out, dim=0)
        assert out.shape == (self.D, )

        out = inputs[-1] + self.acc_factor * out
        assert out.shape == (4, )
        return out


class DirectionObservation(BaseObservation):
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
    def __init__(self, K, D, M=0, mus_init=None, sigmas=None,
                 bounds=None, train_sigma=True, **transformation_kwargs):
        super(DirectionObservation, self).__init__(K, D, M)

        self.momentum_lags = transformation_kwargs.get("momentum_lags", None)
        if self.momentum_lags is None:
            raise ValueError("Must provide momentum lags.")
        assert self.momentum_lags > 1

        self.transformation = DirectionTransformation(K=K, D=D, **transformation_kwargs)

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
            self.mus_init = check_and_convert_to_tensor(mus_init)

        self.log_sigmas_init = torch.tensor(np.log(np.ones((K, D))), dtype=torch.float64)

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
        self.log_sigmas = self.log_sigmas[perm]
        self.transformation.permute(perm)

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



