"""
the code is mainly based on https://github.com/slinderman/ssm
"""

import torch
import numpy as np
import numpy.random as npr

from ssm_ptc.transitions.base_transition import BaseTransition
from ssm_ptc.transitions.stationary_transition import StationaryTransition
from ssm_ptc.transitions.sticky_transition import StickyTransition, InputDrivenTransition
from ssm_ptc.observations.base_observation import BaseObservation
from ssm_ptc.observations.ar_gaussian_observation import ARGaussianObservation
from ssm_ptc.observations.ar_logit_normal_observation import ARLogitNormalObservation
from ssm_ptc.observations.ar_truncated_normal_observation import ARTruncatedNormalObservation
from ssm_ptc.message_passing.primitives import viterbi
from ssm_ptc.message_passing.normalizer import hmmnorm_cython
from ssm_ptc.utils import check_and_convert_to_tensor, set_param, ensure_args_are_lists_of_tensors, get_np

from tqdm import trange

TRANSITION_CLASSES = dict(stationary=StationaryTransition,
                          sticky=StickyTransition,
                          inputdriven=InputDrivenTransition)

OBSERVATION_CLASSES = dict(gaussian=ARGaussianObservation,
                           logitnormal=ARLogitNormalObservation,
                           truncatednormal=ARTruncatedNormalObservation)


class HMM:

    def __init__(self, K, D, M=0, transition='stationary', observation="gaussian", pi0=None, Pi=None,
                 transition_kwargs=None, observation_kwargs=None, device=torch.device('cpu')):
        """
        :param K: number of hidden states
        :param D: dimension of observations
        :param M: dimension of inputs
        """
        assert isinstance(K, int)
        assert isinstance(D, int)
        assert isinstance(M, int)

        self.K = K
        self.D = D
        self.M = M

        # parameter for the softmax distribution
        if pi0 is None:
            self.pi0 = torch.ones(self.K, dtype=torch.float64, requires_grad=True, device=device)
        else:
            self.pi0 = check_and_convert_to_tensor(pi0, dtype=torch.float64, device=device)

        if isinstance(transition, str):
            transition = transition.lower()

            transition_kwargs = transition_kwargs or {}

            if transition not in TRANSITION_CLASSES:
                raise ValueError("Invalid transition model: {}. Please select from {}.".format(
                    transition, list(TRANSITION_CLASSES.keys())))

            self.transition = TRANSITION_CLASSES[transition](self.K, self.D, self.M, Pi, device=device, **transition_kwargs)

        elif isinstance(transition, BaseTransition):
            self.transition = transition
        else:
            raise ValueError("Invalid transition type.")

        if isinstance(observation, str):
            observation = observation.lower()

            observation_kwargs = observation_kwargs or {}

            if observation not in OBSERVATION_CLASSES:
                raise ValueError("Invalid observaiton model: {}. Please select from {}.".format(
                    observation, list(OBSERVATION_CLASSES.keys())))

            self.observation = OBSERVATION_CLASSES[observation](self.K, self.D, self.M, device=device, **observation_kwargs)

        elif isinstance(observation, BaseObservation):
            self.observation = observation
        else:
            raise ValueError("Invalid observation type.")

        self.device = device

    @ensure_args_are_lists_of_tensors
    def initialize(self, datas, inputs=None):
        self.transition.initialize(datas, inputs)
        self.observation.initialize(datas, inputs)

    @property
    def init_dist(self):
        return torch.nn.Softmax(dim=0)(self.pi0)

    def sample_z(self, T):
        # sample the time-invariant markov chain only

        # TODO: may want to expand to other cases
        assert isinstance(self.transition, StationaryTransition), \
            "Sampling the makov chain only supports for stationary transition"

        z = torch.empty(T, dtype=torch.int, device=self.device)
        pi0 = get_np(self.init_dist)
        z[0] = npr.choice(self.K, p=pi0)

        P = get_np(self.transition.stationary_transition_matrix)  # (K, K)
        for t in range(1, T):
            z[t] = npr.choice(self.K, p=P[z[t - 1]])
        return z

    def sample(self, T, prefix=None, input=None, transformation=False, return_np=True, **kwargs):
        """
        Sample synthetic data form from the model.
        :param T: int, the number of time steps to sample
        :param prefix: (z_pre, x_pre), preceding hidden states and observations.
        z_pre: shape (T_pre,)
        x_pre: shape (T_pre, D)
        :return: (z_sample, x_sample)
        z_sample: shape (T,)
        x_sample: shape (T, D)
        """

        if isinstance(self.transition, InputDrivenTransition) and input is None:
            raise ValueError("Please provide input.")

        if input is not None:
            input = check_and_convert_to_tensor(input, device=self.device)

        K = self.K
        D = self.D
        M = self.M

        dtype = torch.float64

        if prefix is None:
            # no prefix is given. Sample the initial state as the prefix
            T_pre = 1
            z = torch.empty(T, dtype=torch.int, device=self.device)
            data = torch.empty((T, D), dtype=dtype, device=self.device)

            # sample the first state from the initial distribution
            pi0 = get_np(self.init_dist)
            z[0] = npr.choice(self.K, p=pi0)
            data[0] = self.observation.sample_x(z[0], data[:0], expectation=transformation, return_np=False, **kwargs)

            # We only need to sample T-1 datapoints now
            T = T - 1
        else:
            # check that the prefix is of the right shape
            z_pre, x_pre = prefix
            assert len(z_pre.shape) == 1
            T_pre = z_pre.shape[0]
            assert x_pre.shape == (T_pre, self.D)

            z_pre = check_and_convert_to_tensor(z_pre, dtype=torch.int, device=self.device)
            x_pre = check_and_convert_to_tensor(x_pre, dtype=dtype, device=self.device)

            # construct the states and data
            z = torch.cat((z_pre, torch.empty(T, dtype=torch.int, device=self.device)))
            assert z.shape == (T_pre + T, )
            data = torch.cat((x_pre, torch.empty((T, D), dtype=dtype, device=self.device)))

        if isinstance(self.transition, StationaryTransition):
            P = get_np(self.transition.stationary_transition_matrix) # (K, K)
            for t in range(T_pre, T_pre + T):
                z[t] = npr.choice(K, p=P[z[t-1]])
                data[t] = self.observation.sample_x(z[t], data[:t], transformation=transformation, return_np=False,
                                                    **kwargs)
        else:
            for t in range(T_pre, T_pre + T):
                P = get_np(self.transition.transition_matrix(data[t-1:t+1], input[t-1:t+1]))
                assert P.shape == (1, self.K, self.K)
                P = torch.squeeze(P)
                assert P.shape == (self.K, self.K)

                z[t] = npr.choice(K, p=P[z[t-1]])
                data[t] = self.observation.sample_x(z[t], data[:t], transformation=transformation, return_np=False,
                                                    **kwargs)

        assert z.requires_grad is False
        assert data.requires_grad is False

        if prefix is None:
            if return_np:
                return get_np(z), get_np(data)
            return z, data
        else:
            if return_np:
                return get_np(z[T_pre:]), get_np(data[T_pre:])
            return z[T_pre:], data[T_pre:]

    def loss(self, data, input=None, **memory_kwargs):
        return -1. * self.log_probability(data, input, **memory_kwargs)

    def log_prior(self):
        return self.transition.log_prior() + self.observation.log_prior()

    @ensure_args_are_lists_of_tensors
    def log_likelihood(self, datas, inputs=None, **memory_kwargs):
        """

        :param datas : x, [(T_1, D), (T_2, D), ..., (T_batch_size, D)]
        :param inputs: [None, None, ..., None] or [(T_1, M), (T_2, M), ..., (T_batch_size, M)]
        :param memory_kwargs: {} or  {m1s: [], m2s: []}, where each value is a list of length batch_size
        :return: log p(x)
        """
        if isinstance(self.transition, InputDrivenTransition) and inputs is None:
            raise ValueError("Please provide input.")

        batch_size = len(datas)

        list_of_memory_kwargs = [{} for _ in range(batch_size)]
        if memory_kwargs != {}:

            for key, val in memory_kwargs.items():
                val = [val] if not isinstance(val, list) else val
                assert len(val) == batch_size, key + " must be a list of length {}".format(batch_size)

                for i in range(batch_size):
                    list_of_memory_kwargs[i][key] = val[i]

        ll = 0
        for data, input, m_kwargs in zip(datas, inputs, list_of_memory_kwargs):
            data = check_and_convert_to_tensor(data, torch.float64, device=self.device)

            T = data.shape[0]
            log_pi0 = torch.nn.LogSoftmax(dim=0)(self.pi0)  # (K, )

            if isinstance(self.transition, StationaryTransition):
                log_P = self.transition.log_stationary_transition_matrix

                assert log_P.shape == (self.K, self.K)
                if T == 1:
                    log_Ps = log_P[None,][:0]
                else:
                    log_Ps = log_P[None,].repeat(T-1, 1, 1)  # (T-1, K, K)
            else:
                # TODO: test this
                Ps = self.transition.transition_matrix(data, input)
                log_Ps = torch.log(Ps)
                assert log_Ps.shape == (T-1, self.K, self.K)

            log_likes = self.observation.log_prob(data, **m_kwargs)  # (T, K)

            ll = ll + hmmnorm_cython(log_pi0, log_Ps, log_likes)

        return ll

    @ensure_args_are_lists_of_tensors
    def log_probability(self, datas, inputs=None, **memory_kwargs):
        return self.log_likelihood(datas, inputs, **memory_kwargs) + self.log_prior()

    @property
    def params(self):
        """
        :return: a tuple of three items, each item is a list of tensors
        """
        return (self.pi0, ), self.transition.params, self.observation.params

    @params.setter
    def params(self, values):
        """only change values, keep requires_grad property"""
        assert type(values) == tuple

        self.pi0 = set_param(self.pi0, values[0][0])
        self.transition.params = values[1]
        self.observation.params = values[2]

    @property
    def params_unpack(self):
        return self.params[0] + self.params[1] + self.params[2]

    @property
    def trainable_params(self):
        """
        :return: the parameters that require grad. maybe helpful for optimization
        """
        params_unpack = self.params_unpack

        out = []
        for p in params_unpack:
            if p.requires_grad:
                out.append(p)
        return out

    # numpy operation
    def most_likely_states(self, data, input=None, **memory_kwargs):
        if isinstance(self.transition, InputDrivenTransition) and input is None:
            raise ValueError("Please provide input.")

        if input is not None:
            input = check_and_convert_to_tensor(input, device=self.device)

        data = check_and_convert_to_tensor(data, device=self.device)
        T = data.shape[0]

        log_pi0 = get_np(self.init_dist)  # (K, )

        if isinstance(self.transition, StationaryTransition):
            log_Ps = self.transition.log_stationary_transition_matrix
            log_Ps = get_np(log_Ps)  # (K, K)
            log_Ps = log_Ps[None,]
        else:
            # TODO: test this
            log_Ps = self.transition.transition_matrix(data, input, log=True)
            log_Ps = get_np(log_Ps)
            assert log_Ps.shape == (T - 1, self.K, self.K)

        log_likes = get_np(self.observation.log_prob(data, **memory_kwargs))
        return viterbi(log_pi0, log_Ps, log_likes)

    def permute(self, perm):
        self.pi0 = torch.tensor(self.pi0[perm], requires_grad=True)
        self.transition.permute(perm)
        self.observation.permute(perm)

    # return np
    def sample_condition_on_zs(self, zs, x0=None, transformation=False, return_np=True, **kwargs):
        """
        Given a z sequence, generate samples condition on this sequence.
        :param zs: (T, )
        :param x0: shape (D,)
        :param return_np: return np.ndarray or torch.tensor
        :return: generated samples (T, D)
        """

        zs = check_and_convert_to_tensor(zs, dtype=torch.int, device=self.device)
        T = zs.shape[0]

        assert T > 0

        xs = torch.zeros((T, self.D), dtype=torch.float64)
        if T == 1:
            if x0 is not None:
                print("Nothing to sample")
                return
            else:
                return self.observation.sample_x(zs[0], expectation=transformation)

        if x0 is None:
            x0 = self.observation.sample_x(zs[0], expectation=transformation, return_np=False)
        else:
            x0 = check_and_convert_to_tensor(x0, dtype=torch.float64, device=self.device)
            assert x0.shape == (self.D, )

        xs[0] = x0
        for t in np.arange(1, T):
            x_t = self.observation.sample_x(zs[t], xihst=xs[:t], expectation=transformation, return_np=False, **kwargs)
            xs[t] = x_t

        if return_np:
            return get_np(xs)
        return xs

    @ensure_args_are_lists_of_tensors
    def fit(self, datas, inputs=None, optimizer=None, method='adam', num_iters=1000, lr=0.001,
            pbar_update_interval=10, valid_data=None, valid_data_memory_kwargs=None, **memory_kwargs):
        # TODO, need to add valid_data to valid_datas
        # TODO: need to modify this
        if isinstance(self.transition, InputDrivenTransition) and inputs is None:
            raise ValueError("Please provide input.")

        pbar = trange(num_iters)

        if optimizer is None:
            if method == 'adam':
                optimizer = torch.optim.Adam(self.trainable_params, lr=lr)
            elif method == 'sgd':
                optimizer = torch.optim.SGD(self.trainable_params, lr=lr)
            else:
                raise ValueError("Method must be chosen from adam and sgd.")

        elif optimizer is not None:
            assert isinstance(optimizer, (torch.optim.SGD, torch.optim.Adam)), \
                "Optimizer must be chosen from SGD or Adam"
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        losses = []
        if valid_data is not None:
            valid_losses = []
            valid_data_memory_kwargs = valid_data_memory_kwargs if valid_data_memory_kwargs else {}
        for i in np.arange(num_iters):
            optimizer.zero_grad()

            loss = self.loss(datas, inputs, **memory_kwargs)
            loss.backward()
            optimizer.step()

            loss = get_np(loss)
            losses.append(loss)

            if valid_data is not None:
                with torch.no_grad():
                    valid_losses.append(get_np(self.loss(valid_data, **valid_data_memory_kwargs)))

            if i % pbar_update_interval == 0:
                pbar.set_description('iter {} loss {:.2f}'.format(i, loss))
                pbar.update(pbar_update_interval)

        pbar.close()

        if valid_data is not None:
            return losses, optimizer, valid_losses

        return losses, optimizer








