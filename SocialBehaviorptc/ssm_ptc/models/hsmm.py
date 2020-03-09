"""
the code is partly based on https://github.com/slinderman/ssm
"""

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
import itertools

import tqdm
from tqdm import trange
import sys

from ssm_ptc.models.hmm import nostdout
from ssm_ptc.init_state_distns import *
from ssm_ptc.observations import *
from ssm_ptc.transitions import *
from ssm_ptc.utils import check_and_convert_to_tensor, ensure_args_are_lists_of_tensors, get_np
from ssm_ptc.message_passing.infc import *

INIT_CLASSES = dict(base=BaseInitStateDistn)
TRANSITION_CLASSES = dict(stationary=StationaryTransition,
                          sticky=StickyTransition,
                          inputdriven=InputDrivenTransition,
                          grid=GridTransition)
OBSERVATION_CLASSES = dict(gaussian=ARGaussianObservation,
                           logitnormal=ARLogitNormalObservation,
                           truncatednormal=ARTruncatedNormalObservation)

class HSMM:
    """
    Hidden semi-Markov model with non-geometric duration distributions.
    The trick is to expand the state space with "super states" and "sub states"
    that effectively count duration. We rely on the transition model to
    specify a "state map," which maps the super states (1, .., K) to
    super+sub states ((1,1), ..., (1,r_1), ..., (K,1), ..., (K,r_K)).
    Here, r_k denotes the number of sub-states of state k.
    """

    def __init__(self, K, D, L, M=0, init_state_distn=None,
                 transition="nb", transition_kwargs=None,
                 observation="gaussian", observation_kwargs=None,
                 device=torch.device('cpu'),
                 **kwargs):

        self.K, self.D, self.M, self.L = K, D, M, L
        self.device = device

        if init_state_distn is None:
            # set to default
            self.init_state_distn = BaseInitStateDistn(K=self.K, D=self.D, M=self.M, device=None)
        elif isinstance(init_state_distn, BaseInitStateDistn):
            self.init_state_distn = init_state_distn
        else:
            raise TypeError("'init_state_distn' must be a subclass of"
                            " ssm.init_state_distns.BaseInitStateDistn")

        # Make the transition model
        if isinstance(transition, str):
            if transition not in TRANSITION_CLASSES:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transition, list(TRANSITION_CLASSES.keys())))

            transition_kwargs = transition_kwargs or {}
            self.transition = TRANSITION_CLASSES[transition](K, D, M=M, **transition_kwargs)
        elif isinstance(transition, BaseTransition):
            self.transition = transition
        else:
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # This is the master list of observation classes.
        # When you create a new observation class, add it here.
        if isinstance(observation, str):
            observations = observation.lower()
            if observations not in OBSERVATION_CLASSES:
                raise Exception("Invalid observation model: {}. Must be one of {}".
                    format(observations, list(OBSERVATION_CLASSES.keys())))

            observation_kwargs = observation_kwargs or {}
            self.observation = OBSERVATION_CLASSES[observations](K, D, M=M, **observation_kwargs)
        elif isinstance(observation, BaseObservation):
            self.observation = observation
        else:
            raise TypeError("'observations' must be a subclass of"
                            " BaseObservation")

    @property
    def params(self):
        return self.init_state_distn.params, self.transition.params, self.observation.params

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

    """
    def params(self):
        result = []
        for object in [self.init_state_distn, self.transition, self.observation]:
            if (object is not None) and isinstance(object, nn.Module):
                result = itertools.chain(result, object.parameters())

        if isinstance(result, list):
            return None
        else:
            return result
    """

    # TODO: implement this
    def sample(self, T, prefix=None, input=None, tag=None, with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).
        Parameters
        ----------
        T : int
            number of time steps to sample
        prefix : (zpre, xpre)
            Optional prefix of discrete states (zpre) and continuous states (xpre)
            zpre must be an array of integers taking values 0...num_states-1.
            xpre must be an array of the same length that has preceding observations.
        input : (T, input_dim) array_like
            Optional inputs to specify for sampling
        tag : object
            Optional tag indicating which "type" of sampled data
        with_noise : bool
            Whether or not to sample data with noise.
        Returns
        -------
        z_sample : array_like of type int
            Sequence of sampled discrete states
        x_sample : (T x observation_dim) array_like
            Array of sampled data
        """
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        assert T > 0

        # Check the inputs
        if input is not None:
            assert input.shape == (T,) + M

        # Get the type of the observations
        dummy_data = self.observation.sample_x(0, np.empty(0,) + D)
        dtype = dummy_data.dtype

        # Initialize the data array
        if prefix is None:
            # No prefix is given.  Sample the initial state as the prefix.
            pad = 1
            z = np.zeros(T, dtype=int)
            data = np.zeros((T,) + D, dtype=dtype)
            input = np.zeros((T,) + M) if input is None else input
            mask = np.ones((T,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = self.init_state_distn.initial_state_distn
            z[0] = npr.choice(self.K, p=pi0)
            data[0] = self.observation.sample_x(z[0], data[:0], input=input[0], with_noise=with_noise)

            # We only need to sample T-1 datapoints now
            T = T - 1

        else:
            # Check that the prefix is of the right type
            zpre, xpre = prefix
            pad = len(zpre)
            assert zpre.dtype == int and zpre.min() >= 0 and zpre.max() < K
            assert xpre.shape == (pad,) + D

            # Construct the states, data, inputs, and mask arrays
            z = np.concatenate((zpre, np.zeros(T, dtype=int)))
            data = np.concatenate((xpre, np.zeros((T,) + D, dtype)))
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            mask = np.ones((T+pad,) + D, dtype=bool)

        # Convert the discrete states to the range (1, ..., K_total)
        m = self.state_map
        K_total = len(m)
        _, starts = np.unique(m, return_index=True)
        z = starts[z]

        # Fill in the rest of the data
        for t in range(pad, pad+T):
            Pt = self.transition.transition_matrices(data[t-1:t+1], input[t-1:t+1], mask=mask[t-1:t+1], tag=tag)[0]
            z[t] = npr.choice(K_total, p=Pt[z[t-1]])
            data[t] = self.observation.sample_x(m[z[t]], data[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Collapse the states
        z = m[z]

        # Return the whole data if no prefix is given.
        # Otherwise, just return the simulated part.
        if prefix is None:
            return z, data
        else:
            return z[pad:], data[pad:]

    # TODO: avoid unnecessary computation
    # TODO: implement viterbi
    def most_likely_states(self, data, input=None, cache=None, transition_mkwargs=None, **memory_kwargs):
        if len(data) == 0:
            return np.array([])

        log_pi0 = cache.get("log_pi0", None)
        log_Ps = cache.get("log_Ps", None)
        bwd_obs_logprobs = cache.get("bwd_obs_log_probs", None)

        if input is not None:
            input = check_and_convert_to_tensor(input, device=self.device)

        data = check_and_convert_to_tensor(data, device=self.device)
        T = data.shape[0]

        if log_pi0 is None:
            log_pi0 = get_np(self.init_dist)  # (K, )

        if log_Ps is None:
            if isinstance(self.transition, StationaryTransition):
                log_Ps = self.transition.log_stationary_transition_matrix
                log_Ps = get_np(log_Ps)  # (K, K)
                log_Ps = log_Ps[None,]
            else:
                assert isinstance(self.transition, GridTransition), type(self.transition)
                transition_mkwargs = transition_mkwargs if transition_mkwargs else {}
                input = input[:-1] if input else input
                log_Ps = self.transition.log_transition_matrix(data[:-1], input, **transition_mkwargs)
                log_Ps = get_np(log_Ps)
                assert log_Ps.shape == (T - 1, self.K, self.K), log_Ps.shape
        if bwd_obs_logprobs is None:
            log_likes = get_np(self.observation.log_prob(data, **memory_kwargs))



        return hsmm_viterbi(log_pi0, trans_logprobs=log_Ps, bwd_obs_logprobs=bwd_obs_logprobs, len_logprobs=None)

    @ensure_args_are_lists_of_tensors
    def fit(self, datas, inputs=None, optimizer=None, method='adam', num_iters=1000, lr=0.001,
            pbar_update_interval=10, valid_data=None,
            transition_memory_kwargs=None, valid_data_transition_memory_kwargs=None,
            valid_data_memory_kwargs=None, **memory_kwargs):
        pbar = trange(num_iters, file=sys.stdout)

        if optimizer is None:
            if method == 'adam':
                optimizer = torch.optim.Adam(self.trainable_params, lr=lr)
            elif method == 'sgd':
                optimizer = torch.optim.SGD(self.trainable_params, lr=lr)
            else:
                raise ValueError("Method must be chosen from adam and sgd.")
        else:
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
            loss = self.loss(datas, inputs, transition_memory_kwargs=transition_memory_kwargs, **memory_kwargs)
            loss.backward()
            optimizer.step()

            loss = get_np(loss)
            losses.append(loss)

            if valid_data is not None:
                if len(valid_data) > 0:
                    with torch.no_grad():
                        valid_losses.append(get_np(
                            self.loss(valid_data, transition_memory_kwargs=valid_data_transition_memory_kwargs,
                                      **valid_data_memory_kwargs)))

            if i % pbar_update_interval == 0:
                with nostdout():
                    pbar.set_description('iter {} loss {:.2f}'.format(i, loss))
                    pbar.update(pbar_update_interval)
        pbar.close()

        if valid_data is not None:
            return losses, optimizer, valid_losses
        return losses, optimizer

    def loss(self, data, input=None, transition_memory_kwargs=None, **memory_kwargs):
        return -1. * self.log_probability(data, input,
                                          transition_memory_kwargs=transition_memory_kwargs, **memory_kwargs)

    def log_prior(self):
        return self.transition.log_prior() + self.observation.log_prior()

    def log_probability(self, datas, inputs=None, transition_memory_kwargs=None, **memory_kwargs):
        return self.log_likelihood(datas, inputs, transition_memory_kwargs=transition_memory_kwargs, **memory_kwargs) + self.log_prior()

    @ensure_args_are_lists_of_tensors
    def log_likelihood(self, datas, inputs=None, transition_memory_kwargs=None, **memory_kwargs):
        """
        Compute the log probability of the data under the current
        model parameters.
        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """

        batch_size = len(datas)

        list_of_transition_mkwargs = [{} for _ in range(batch_size)]
        if transition_memory_kwargs:
            assert isinstance(transition_memory_kwargs, dict), type(transition_memory_kwargs)
            for key, val in transition_memory_kwargs.items():
                # TODO: some ad-hoc fix
                if isinstance(val, list) and not isinstance(val[0], list):
                    val = [val]
                assert len(val) == batch_size, key + " must be a list of length {}".format(batch_size)
                for i in range(batch_size):
                    list_of_transition_mkwargs[i][key] = val[i]

        list_of_memory_kwargs = [{} for _ in range(batch_size)]
        if memory_kwargs != {}:
            for key, val in memory_kwargs.items():
                val = [val] if not isinstance(val, list) else val
                assert len(val) == batch_size, key + " must be a list of length {}".format(batch_size)
                for i in range(batch_size):
                    list_of_memory_kwargs[i][key] = val[i]

        ll = 0
        for data, input, transition_mkwargs, m_kwargs \
                in zip(datas, inputs, list_of_transition_mkwargs, list_of_memory_kwargs):
            if len(data) == 0:
                continue
            data = check_and_convert_to_tensor(data, torch.float64, device=self.device)

            T = data.shape[0]
            log_pi0 = self.init_state_distn.log_pi

            if isinstance(self.transition, StationaryTransition):
                log_P = self.transition.log_stationary_transition_matrix
                assert log_P.shape == (self.K, self.K)
                if T == 1:  # TODO: check this
                    log_Ps = log_P[None,][:0]
                else:
                    log_Ps = log_P[None,].repeat(T - 1, 1, 1)  # (T-1, K, K)
            else:
                assert isinstance(self.transition, GridTransition)
                input = input[:-1] if input else input
                log_Ps = self.transition.log_transition_matrix(data[:-1], input, **transition_mkwargs)
                assert log_Ps.shape == (T - 1, self.K, self.K), \
                    "correct shape is {}, but got {}".format((T - 1, self.K, self.K), log_Ps.shape)

            log_likes = self.observation.log_prob(data, **m_kwargs) # (T, K)
            assert log_likes.shape == (T, self.K)
            log_likes = self.stacked_fw_log_likes_helper(log_likes, self.L)

            ll = ll + hsmm_normalizer(log_pi=log_pi0, tran_log_probs=log_Ps, seq_logprobs=None, fwd_obs_logprobs=log_likes)
        return ll

    @staticmethod
    def stacked_fw_log_likes_helper(log_likes, L):
        """

        :param log_likes: (T, K): for each k,  it is like [p1, ..., pT]
        --> [[p1, p2, ..., pT]
            [p1:2, p2:3, ..., p_{T-1:T}, -inf]
            ...
            [p1:L, p2:L+2, ...,p_{L-T+1:T}, -inf]]
        :return: (L, T, K)
        """
        T, K = log_likes.shape
        max_L = min(T, L)

        stacked_log_likes = [log_likes]  # should be (max_L, T, K) in the end
        for l in range(2, max_L+1):
            log_like_l = stacked_log_likes[l-1-1][:T-l+1] + log_likes[l-1:]  # (T-l+1)
            assert log_like_l.shape == (T-l+1, K)
            log_like_l_pad = log_like_l.new(l-1, K).fill_(-float("inf"))
            log_like_l = torch.cat((log_like_l, log_like_l_pad), dim=0) # (T, K)
            assert log_like_l.shape == (T, K), log_like_l.shape
            stacked_log_likes.append(log_like_l)

        stacked_log_likes = torch.stack(stacked_log_likes, dim=0)  # (L, T, K)

        assert stacked_log_likes.shape == (max_L, T, K)
        return stacked_log_likes

    @staticmethod
    def stacked_bw_log_likes_helper(log_likes, L):
        """
        redefine L = min(T, K)
        :param log_likes: (T, K): for each k, it is like [p1, ..., pT]
        --> [[p1, p2, ..., pT]
            [-inf, p1:2, ..., p_{T-2:T-1}, p_{T-1:T}]
            ...
            [-inf, ..., -inf, p_{1:L}, p_{T-L+1:T}]]
        :param L:
        :return:
        """
        T, K = log_likes.shape
        max_L = min(T, L)

        stacked_log_likes = [log_likes]  #should be (max_L, T, K) in the end
        for l in range(2, max_L+1):
            log_like_l = stacked_log_likes[l-1-1][:T-l+1] + log_likes[l-1:]  # (T-l+1)
            assert log_like_l.shape == (T-l+1, K)
            log_like_l_pad = log_like_l.new(l-1, K).fill(-float("inf"))
            log_like_l = torch.cat((log_like_l_pad, log_like_l), dim=0)  # (T, K)
            assert log_like_l.shape == (T, K), log_like_l.shape
            stacked_log_likes.append(log_like_l)

        stacked_log_likes = torch.stack(stacked_log_likes, dim=0)  # (L, T, K)
        assert stacked_log_likes.shape == (max_L, T, K)
        return stacked_log_likes

    @staticmethod
    def stacked_log_likes_helper_over_T(log_likes, L):
        T, K = log_likes.shape
        stacked_log_likes = []  # should be (L, T, K) in the end
        # fill in by column (the second dimension)

        max_L = min(L, T)

        for t in range(T):
            if t < T-max_L:
                l_obs_t = torch.cumsum(log_likes[t:t + max_L], dim=0)  # (max_L, K)
            else:
                l_obs_t = torch.cumsum(log_likes[t:], dim=0)  # (T-t, L), T-t < max_L-2
                l_obs_t = torch.cat((l_obs_t, l_obs_t.new(max_L - (T-t), K).fill_(-float("inf"))),
                                                 dim=0)  # (max_L, K)
            stacked_log_likes.append(l_obs_t)
            assert stacked_log_likes[t].shape == (max_L, K), stacked_log_likes[t].shape

        stacked_log_likes = torch.stack(stacked_log_likes, dim=1)
        assert stacked_log_likes.shape == (max_L, T, K)
        return stacked_log_likes

    @staticmethod
    def test_stacked_log_likes():
        p1_k1, p2_k1, p3_k1, p4_k1 = 1.5, 2.5, 3, 4
        p1_k2, p2_k2, p3_k2, p4_k2 = 1.3, 2.4, 2.5, 6.6
        log_likes = torch.tensor([[p1_k1, p1_k2], [p2_k1, p2_k2], [p3_k1, p3_k2], [p4_k1, p4_k2]])

        # L<T
        T = 4
        L = 3

        true_stacked_log_likes_k1 = torch.tensor([[p1_k1, p2_k1, p3_k1, p4_k1],
                                                 [p1_k1+p2_k1, p2_k1+p3_k1, p3_k1+p4_k1, -float("inf")],
                                                  [p1_k1+p2_k1+p3_k1, p2_k1+p3_k1+p4_k1, -float("inf"), -float("inf")]])
        true_stacked_log_likes_k2 = torch.tensor([[p1_k2, p2_k2, p3_k2, p4_k2],
                                                  [p1_k2+p2_k2, p2_k2+p3_k2, p3_k2+p4_k2, -float("inf")],
                                                  [p1_k2+p2_k2+p3_k2, p2_k2+p3_k2+p4_k2, -float("inf"), -float("inf")]])
        true_stacked_log_likes = torch.stack([true_stacked_log_likes_k1, true_stacked_log_likes_k2], dim=2) # (L, T, K)

        s_over_L = HSMM.stacked_fw_log_likes_helper(log_likes=log_likes, L=L)
        assert torch.all(true_stacked_log_likes == s_over_L), "true = \n{}\n computed = \n{}".format(true_stacked_log_likes, s_over_L)

        print("\n")

        s_over_T = HSMM.stacked_log_likes_helper_over_T(log_likes=log_likes, L=L)
        assert torch.all(true_stacked_log_likes == s_over_T), "true = \n{}\n computed = \n{}".format(
            true_stacked_log_likes, s_over_T)

        #  L >= T
        L = 5
        l4_probs = torch.tensor([[p1_k1+p2_k1+p3_k1+p4_k1, p1_k2+p2_k2+p3_k2+p4_k2],
                                 [-float("inf"), -float("inf")],
                                 [-float("inf"), -float("inf")],
                                 [-float("inf"), -float("inf")]])  # (T, K)
        true_stacked_log_likes = torch.cat((true_stacked_log_likes, l4_probs[None, ])) # (L, T, K)

        s_over_L = HSMM.stacked_fw_log_likes_helper(log_likes=log_likes, L=L)
        assert torch.allclose(true_stacked_log_likes, s_over_L), "true = \n{}\n computed = \n{}".format(
            true_stacked_log_likes, s_over_L)

        print("\n")

        s_over_T = HSMM.stacked_log_likes_helper_over_T(log_likes=log_likes, L=L)
        assert torch.allclose(true_stacked_log_likes, s_over_T), "true = \n{}\n computed = \n{}".format(
            true_stacked_log_likes, s_over_T)


if __name__ == "__main__":
    #HSMM.test_stacked_log_likes()
    torch.random.manual_seed(0)
    np.random.seed(0)

    import git
    import joblib

    from project_ssms.utils import downsample
    from project_ssms.gp_observation_single import GPObservationSingle
    from project_ssms.constants import *

    # test for virgin selected
    repo = git.Repo('.', search_parent_directories=True)  # SocialBehaviorectories=True)
    repo_dir = repo.working_tree_dir  # SocialBehavior

    start = 0
    end = 1

    data_dir = repo_dir + '/SocialBehaviorptc/data/traj_010_virgin_selected'
    traj = joblib.load(data_dir)
    T = len(traj)
    traj = traj[int(T * start): int(T * end)]

    downsample_n = 4
    traj = downsample(traj, downsample_n)
    traj = traj[200:500]

    device = torch.device('cpu')
    data = torch.tensor(traj, dtype=torch.float64, device=device)

    K = 2
    T, D = data.shape

    n_x = 3
    n_y = 3
    mus_init = data[0] * torch.ones(K, D, dtype=torch.float64, device=device)
    x_grid_gap = (ARENA_XMAX - ARENA_XMIN) / n_x
    x_grids = np.array([ARENA_XMIN + i * x_grid_gap for i in range(n_x + 1)])
    y_grid_gap = (ARENA_XMAX - ARENA_XMIN) / n_y
    y_grids = np.array([ARENA_XMIN + i * y_grid_gap for i in range(n_y + 1)])
    bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])
    train_rs = False
    obs = GPObservationSingle(K=K, D=D, mus_init=mus_init, x_grids=x_grids, y_grids=y_grids, bounds=bounds,
                              rs=None, train_rs=train_rs, device=device)

    L = 5
    model = HSMM(K=K, D=D, L=L, init_state_distn=None, transition='stationary', observation=obs)

    ll = model.log_likelihood(data)  # tensor(-10207.9570, grad_fn=<AddBackward0>)
    print(ll)

    # fitting
    loss = model.fit(datas=data, num_iters=100, lr=1e-3)  # 8744.38





