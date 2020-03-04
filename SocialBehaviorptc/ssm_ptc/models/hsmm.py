"""
the code is mainly based on https://github.com/slinderman/ssm
"""

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
import itertools

from ssm_ptc.models.hmm import HMM
from ssm_ptc.init_state_distns import *
from ssm_ptc.observations import *
from ssm_ptc.transitions import *
from ssm_ptc.message_passing.primitives import viterbi
from ssm_ptc.message_passing.normalizer import hmmnorm_cython

INIT_CLASSES = dict(base=BaseInitStateDistn)
TRANSITION_CLASSES = dict(nb=NegativeBinomialSemiMarkovTransition)
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

    def __init__(self, K, D, *, M=0, init_state_distn=None,
                 transition="nb", transition_kwargs=None,
                 observation="gaussian", observation_kwargs=None,
                 **kwargs):

        if init_state_distn is None:
            # set to default
            pass
            #init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        if not isinstance(init_state_distn, BaseInitStateDistn):
            raise TypeError("'init_state_distn' must be a subclass of"
                            " ssm.init_state_distns.BaseInitStateDistn")

        # Make the transition model
        if isinstance(transition, str):
            if transition not in TRANSITION_CLASSES:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transition, list(TRANSITION_CLASSES.keys())))

            transition_kwargs = transition_kwargs or {}
            transition = TRANSITION_CLASSES[transition](K, D, M=M, **transition_kwargs)
        if not isinstance(transition, BaseTransition):
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
        if not isinstance(observation, BaseObservation):
            raise TypeError("'observations' must be a subclass of"
                            " BaseObservation")

        self.K = K
        self.D = D
        self.M = M
        self.init_state_distn = INIT_CLASSES[init_state_distn]
        self.transition = TRANSITION_CLASSES[transition]
        self.observation = OBSERVATION_CLASSES[observation]

    @property
    def params(self):
        result = []
        for object in [self.init_state_distn, self.transition, self.observation]:
            if (object is not None) and isinstance(object, nn.Module):
                result = itertools.chain(result, object.parameters())

        if isinstance(result, list):
            return None
        else:
            return result

    @property
    def state_map(self):
        return self.transition.state_map

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

    def most_likely_states(self, data, input=None, mask=None, tag=None):
        m = self.state_map
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        z_star = viterbi(replicate(pi0, m), Ps, replicate(log_likes, m))
        return self.state_map[z_star]

    def filter(self, data, input=None, mask=None, tag=None):
        m = self.state_map
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        pzp1 = hmm_filter(replicate(pi0, m), Ps, replicate(log_likes, m))
        return collapse(pzp1, m)

    def posterior_sample(self, data, input=None, mask=None, tag=None):
        m = self.state_map
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transition.transition_matrices(data, input, mask, tag)
        log_likes = self.observation.log_likelihoods(data, input, mask, tag)
        z_smpl = hmm_sample(replicate(pi0, m), Ps, replicate(log_likes, m))
        return self.state_map[z_smpl]

    def smooth(self, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        m = self.state_map
        Ez, _, _ = self.expected_states(data, input, mask)
        return self.observation.smooth(Ez, data, input, tag)

    def log_likelihood(self, datas, inputs=None, **memory_kwargs):
        """
        Compute the log probability of the data under the current
        model parameters.
        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        m = self.state_map
        ll = 0
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            init_logps = self.init_state_distn.initial_state_distn
            # assume to be staitonary for now
            trans_log_probs = self.transition.log_stationary_matrix
            # assume to be uniform now
            #len_log_probs = self.len_log_probs()
            log_likes = self.observation.log_likelihoods(data, input, )

            """
            pi0 = self.init_state_distn.initial_state_distn
            Ps = self.transition.transition_matrices(data, input, mask, tag)
            log_likes = self.observation.log_likelihoods(data, input, mask, tag)
            ll += hmm_normalizer(replicate(pi0, m), Ps, replicate(log_likes, m))
            assert np.isfinite(ll)
            """
        return ll

    def len_log_probs(self):


    def expected_log_probability(self, expectations, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log probability of the data under the current
        model parameters.
        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        raise NotImplementedError("Need to get raw expectations for the expected transition probability.")
