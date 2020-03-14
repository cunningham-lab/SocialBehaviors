"""
the code is mainly based on https://github.com/slinderman/ssm
"""

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
import sys
import contextlib
import itertools

from ssm_ptc.init_state_distns import *
from ssm_ptc.observations import *
from ssm_ptc.transitions import *
from ssm_ptc.message_passing.primitives import viterbi
from ssm_ptc.message_passing.normalizer import hmmnorm_cython
from ssm_ptc.utils import check_and_convert_to_tensor, set_param, ensure_args_are_lists_of_tensors, get_np

import tqdm
from tqdm import trange
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from project_ssms.plot_utils import get_colors_and_cmap, add_grid


INIT_CLASSES = dict(base=BaseInitStateDistn)
TRANSITION_CLASSES = dict(stationary=StationaryTransition,
                          sticky=StickyTransition,
                          inputdriven=InputDrivenTransition,
                          grid=GridTransition)

OBSERVATION_CLASSES = dict(gaussian=ARGaussianObservation,
                           truncatednormal=ARTruncatedNormalObservation)


class DummyFile(object):
  file = None
  def __init__(self, file):
    self.file = file

  def write(self, x):
    # Avoid print() second call (useless \n)
    if len(x.rstrip()) > 0:
        tqdm.write(x, file=self.file)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


class HMM:

    def __init__(self, K, D, M=0, init_state_distn=None, transition='stationary', observation="gaussian",Pi=None,
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

        if init_state_distn is None:
            # set to default
            self.init_state_distn = BaseInitStateDistn(K=self.K, D=self.D, M=self.M)
        elif isinstance(init_state_distn, BaseInitStateDistn):
            self.init_state_distn = init_state_distn
        else:
            raise TypeError("'init_state_distn' must be a subclass of"
                            " ssm.init_state_distns.BaseInitStateDistn")

        if isinstance(transition, str):
            transition = transition.lower()
            transition_kwargs = transition_kwargs or {}
            if transition not in TRANSITION_CLASSES:
                raise ValueError("Invalid transition model: {}. Please select from {}.".format(
                    transition, list(TRANSITION_CLASSES.keys())))

            self.transition = TRANSITION_CLASSES[transition](self.K, self.D, self.M, Pi=Pi, **transition_kwargs)

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

            self.observation = OBSERVATION_CLASSES[observation](self.K, self.D, self.M, **observation_kwargs)

        elif isinstance(observation, BaseObservation):
            self.observation = observation
        else:
            raise ValueError("Invalid observation type.")

        self.device = device

    @ensure_args_are_lists_of_tensors
    def initialize(self, datas, inputs=None):
        self.transition.initialize(datas, inputs)
        self.observation.initialize(datas, inputs)

    def sample_z(self, T):
        # sample the downsampled_t-invariant markov chain only

        # TODO: may want to expand to other cases
        assert isinstance(self.transition, StationaryTransition), \
            "Sampling the makov chain only supports for stationary transition"

        z = torch.empty(T, dtype=torch.int, device=self.device)
        pi0 = get_np(self.init_state_distn.probs)
        z[0] = npr.choice(self.K, p=pi0)

        P = get_np(self.transition.stationary_transition_matrix)  # (K, K)
        for t in range(1, T):
            z[t] = npr.choice(self.K, p=P[z[t - 1]])
        return z

    def sample(self, T, prefix=None, input=None, with_noise=True, return_np=True, **kwargs):
        """
        Sample synthetic data form from the model.
        :param T: int, the number of downsampled_t steps to sample
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

        dtype = torch.float32

        if prefix is None:
            # no prefix is given. Sample the initial state as the prefix
            T_pre = 1
            z = torch.empty(T, dtype=torch.int, device=self.device)
            data = torch.empty((T, D), dtype=dtype, device=self.device)

            # sample the first state from the initial distribution
            pi0 = get_np(self.init_state_distn.probs)
            z[0] = npr.choice(self.K, p=pi0)
            data[0] = self.observation.sample_x(z[0], data[:0], with_noise=with_noise, return_np=False, **kwargs)

            # We only need to sample T-1 datapoints now
            T = T - 1
        else:
            # check that the prefix is of the right shape
            z_pre, x_pre = prefix
            assert len(z_pre.shape) == 1
            T_pre = z_pre.shape[0]
            assert x_pre.shape == (T_pre, self.D), "should be {}, but got {}.".format((T_pre, self.D), x_pre.shape)

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
                data[t] = self.observation.sample_x(z[t], data[:t], with_noise=with_noise, return_np=False,
                                                    **kwargs)
        else:
            for t in range(T_pre, T_pre + T):
                input_t = input[t-1:t] if input else input
                P = self.transition.transition_matrix(data[t-1:t], input_t)
                assert P.shape == (1, self.K, self.K)
                P = torch.squeeze(P, dim=0)
                P = get_np(P)

                z[t] = npr.choice(K, p=P[z[t-1]])
                data[t] = self.observation.sample_x(z[t], data[:t], with_noise=with_noise, return_np=False,
                                                    **kwargs)
        #assert not z.requires_grad
        #assert not data.requires_grad

        if prefix is None:
            if return_np:
                return get_np(z), get_np(data)
            return z, data
        else:
            if return_np:
                return get_np(z[T_pre:]), get_np(data[T_pre:])
            return z[T_pre:], data[T_pre:]

    def loss(self, data, input=None, transition_memory_kwargs=None, **memory_kwargs):
        return -1. * self.log_probability(data, input,
                                          transition_memory_kwargs=transition_memory_kwargs, **memory_kwargs)

    def log_prior(self):
        return self.transition.log_prior() + self.observation.log_prior()

    @ensure_args_are_lists_of_tensors
    def log_likelihood(self, datas, inputs=None, transition_memory_kwargs=None, **memory_kwargs):
        """

        :param datas : x, [(T_1, D), (T_2, D), ..., (T_batch_size, D)]
        :param inputs: [None, None, ..., None] or [(T_1, M), (T_2, M), ..., (T_batch_size, M)]
        :param memory_kwargs: {} or  {m1s: [], m2s: []}, where each value is a list of length batch_size
        :return: log p(x)
        """
        if isinstance(self.transition, InputDrivenTransition) and inputs is None:
            raise ValueError("Please provide input.")

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
            log_pi0 = self.init_state_distn.log_probs

            if isinstance(self.transition, StationaryTransition):
                log_P = self.transition.log_stationary_transition_matrix
                assert log_P.shape == (self.K, self.K)
                if T == 1: # TODO: check this
                    log_Ps = log_P[None,][:0]
                else:
                    log_Ps = log_P[None,].repeat(T-1, 1, 1)  # (T-1, K, K)
            else:
                assert isinstance(self.transition, GridTransition)
                input = input[:-1] if input else input
                log_Ps = self.transition.log_transition_matrix(data[:-1], input, **transition_mkwargs)
                assert log_Ps.shape == (T-1, self.K, self.K), \
                    "correct shape is {}, but got {}".format((T-1, self.K, self.K), log_Ps.shape)

            log_likes = self.observation.log_prob(data, **m_kwargs)  # (T, K)

            ll = ll + hmmnorm_cython(log_pi0, log_Ps, log_likes)

        return ll

    def log_probability(self, datas, inputs=None, transition_memory_kwargs=None, **memory_kwargs):
        return self.log_likelihood(datas, inputs, transition_memory_kwargs=transition_memory_kwargs, **memory_kwargs) + self.log_prior()

    @property
    def params(self):
        result = []
        for object in [self.init_state_distn, self.transition, self.observation]:
            if (object is not None) and isinstance(object, nn.Module):
                result = itertools.chain(result, object.parameters())
        return result

    @property
    def trainable_params(self):
        result = []
        for object in [self.init_state_distn, self.transition, self.observation]:
            if (object is not None) and isinstance(object, nn.Module):
                result = itertools.chain(result, filter(lambda p: p.requires_grad, object.parameters()))

        return result

    # numpy operation
    def most_likely_states(self, data, input=None, transition_mkwargs=None, **memory_kwargs):
        if len(data) == 0:
            return np.array([])
        if isinstance(self.transition, InputDrivenTransition) and input is None:
            raise ValueError("Please provide input.")

        if input is not None:
            input = check_and_convert_to_tensor(input, device=self.device)

        data = check_and_convert_to_tensor(data, device=self.device)
        T = data.shape[0]

        log_pi0 = get_np(self.init_state_distn.log_probs)

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

        log_likes = get_np(self.observation.log_prob(data, **memory_kwargs))
        return viterbi(log_pi0, log_Ps, log_likes)

    def permute(self, perm):
        with torch.no_grad():
            self.init_state_distn.permute(perm)
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
        dtype = torch.float64
        xs = torch.zeros((T, self.D), dtype=dtype)
        if T == 1:
            if x0 is not None:
                print("Nothing to sample")
                return
            else:
                return self.observation.sample_x(zs[0], with_noise=transformation)

        if x0 is None:
            x0 = self.observation.sample_x(zs[0], with_noise=transformation, return_np=False)
        else:
            x0 = check_and_convert_to_tensor(x0, dtype=dtype, device=self.device)
            assert x0.shape == (self.D, )

        xs[0] = x0
        for t in np.arange(1, T):
            x_t = self.observation.sample_x(zs[t], xihst=xs[:t], with_noise=transformation, return_np=False, **kwargs)
            xs[t] = x_t

        if return_np:
            return get_np(xs)
        return xs

    @ensure_args_are_lists_of_tensors
    def fit(self, datas, inputs=None, optimizer=None, method='adam', num_iters=1000, lr=0.001,
            pbar_update_interval=10, valid_data=None,
            transition_memory_kwargs=None, valid_data_transition_memory_kwargs=None,
            valid_data_memory_kwargs=None, **memory_kwargs):
        plot_color_quiver = False
        plot_dynamics_quiver = False
        plot_transition = False
        plot_update_interval = 5

        # TODO, need to add valid_data to valid_datas
        # TODO: need to modify this
        if isinstance(self.transition, InputDrivenTransition) and inputs is None:
            raise ValueError("Please provide input.")

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
        # TODO: collect rs. delete later
        if isinstance(self.observation, GPObservationSingle):
            obs_rs = [get_np(self.observation.rs.clone())]
            x_grids = self.observation.x_grids
            y_grids = self.observation.y_grids
        else:
            print("observation tye", type(self.observation))
            x_grids = self.observation.transformation.x_grids
            y_grids = self.observation.transformation.y_grids
            obs_rs = []

        K = self.K
        if plot_color_quiver:
            h = 1 / K
            ticks = [(1 / 2 + k) * h for k in range(K)]
            colors, cm = get_colors_and_cmap(K)
            fig1 = plt.figure(figsize=(8, 7))
            ax1 = fig1.add_subplot(1,1,1)
            self.plot_realdata_quiver(0, "init", None, datas[0], colors, cm, transition_memory_kwargs, memory_kwargs)
            cb = plt.colorbar(label='k', ticks=ticks)
            cb.set_ticklabels(range(K))
            add_grid(x_grids=x_grids, y_grids=y_grids)
            plt.pause(0.002)

        if plot_dynamics_quiver:
            fig, axs = plt.subplots(nrows=1, ncols=K, figsize=(4 * K, 4))
            self.plot_quiver(axs=axs, K=K, scale=1, alpha=1,
                             x_grids=x_grids, y_grids=y_grids, grid_alpha=1)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle("iter {}".format(0))
            plt.pause(0.002)
        if plot_transition:
            if isinstance(self.transition, StationaryTransition):
                fig2 = plt.figure(figsize=(6,6))
                ax_transition = fig2.add_subplot(1,1,1)
                cbar_ax = fig2.add_axes([.93, .3, .03, .4])
                self.plot_transition(ax_transition, cbar_ax, get_np(self.transition.stationary_transition_matrix))
            elif isinstance(self.transition, GridTransition):
                n_x = len(x_grids) - 1
                n_y = len(y_grids) - 1
                fig2, axn = plt.subplots(n_x, n_y, sharex=True, sharey=True, figsize=(6, 6))
                cbar_ax = fig2.add_axes([.93, .3, .03, .4])
                self.plot_grid_transition(axn=axn, cbar_ax=cbar_ax, n_x=n_x, n_y=n_y,
                                          grid_transition=get_np(self.transition.grid_transition_matrix))
                fig2.tight_layout(rect=[0, 0, .9, 1])
            fig2.suptitle("iter {}".format(0))
            plt.pause(0.002)

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

            # TODO: collect rs. delete later
            if isinstance(self.observation, GPObservationSingle):
                obs_rs.append(get_np(self.observation.rs.clone()))

            if i % plot_update_interval == 0:
                if plot_color_quiver:
                    self.plot_realdata_quiver(i, loss,ax1, datas[0], colors, cm, transition_memory_kwargs, memory_kwargs)
                    plt.pause(0.001)
                if plot_dynamics_quiver:
                    self.plot_quiver(axs=axs, K=K, scale=1, alpha=1,
                                     x_grids=x_grids, y_grids=y_grids, grid_alpha=1)
                    fig.suptitle("iter {}".format(i+1))
                    plt.pause(0.001)
                if plot_transition:
                    fig2.suptitle("iter {}".format(i + 1))
                    if isinstance(self.transition, StationaryTransition):
                        self.plot_transition(ax_transition, cbar_ax, get_np(self.transition.stationary_transition_matrix))
                    elif isinstance(self.transition, GridTransition):
                        cbar_ax = fig2.add_axes([.93, .3, .03, .4])
                        self.plot_grid_transition(axn=axn, cbar_ax=cbar_ax, n_x=n_x, n_y=n_y,
                                                  grid_transition=get_np(self.transition.grid_transition_matrix))
                    plt.pause(0.001)


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
            return losses, optimizer, valid_losses, obs_rs

        return losses, optimizer, obs_rs

    def plot_realdata_quiver(self, i, loss, ax, realdata, colors, cm, transition_memory_kwargs, memory_kwargs, **quiver_args):
        z = self.most_likely_states(realdata, transition_mkwargs=transition_memory_kwargs, **memory_kwargs)
        z = z[1:]  # (T-1, )
        start = realdata[:-1]
        end = realdata[1:]
        dXY = end - start
        if ax is None:
            plt.quiver(start[:, 0], start[:, 1], dXY[:, 0], dXY[:, 1],
                      angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[z], **quiver_args)
            plt.title("iter {}, loss = {}".format(i+1, loss))
        else:
            ax.quiver(start[:, 0], start[:, 1], dXY[:, 0], dXY[:, 1],
                       angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[z], **quiver_args)
            ax.set_title("iter {}, loss = {}".format(i+1, loss))

    def plot_quiver(self, axs, K, scale=1, alpha=1, x_grids=None, y_grids=None, grid_alpha=1):
        #TODO: for single animal now
        # quiver
        XX, YY = np.meshgrid(np.linspace(20, 310, 30),
                             np.linspace(0, 380, 30))
        XYs = np.column_stack((np.ravel(XX), np.ravel(YY)))  # shape (900,2) grid values
        if isinstance(self.observation, GPObservationSingle):
            XY_next, _ = self.observation.get_mu_and_cov_for_single_animal(XYs, mu_only=True)
        else:
            XY_next = self.observation.transformation.transform(torch.tensor(XYs, dtype=torch.float64,
                                                                             device=self.device))
        dXYs = get_np(XY_next) - XYs[:, None]

        if K == 1:
            axs.clear()
            axs.quiver(XYs[:, 0], XYs[:, 1], dXYs[:, 0, 0], dXYs[:, 0, 1],
                          angles='xy', scale_units='xy', scale=scale, alpha=alpha)
            # add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
            if isinstance(self.observation, GPObservationSingle):
                axs.set_title("K={}, rs={}".format(0, get_np(self.observation.rs[0])))
            else:
                axs.set_title("K={}".format(0))
        else:
            for k in range(K):
                axs[k].clear()
                axs[k].quiver(XYs[:, 0], XYs[:, 1], dXYs[:, k, 0], dXYs[:, k, 1],
                           angles='xy', scale_units='xy', scale=scale, alpha=alpha)
                #add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
                if isinstance(self.observation, GPObservationSingle):
                    axs[k].set_title('K={}, rs={}'.format(k, get_np(self.observation.rs[k])))
                else:
                    axs[k].set_title('K={}'.format(k))

    def plot_grid_transition(self, axn, cbar_ax, n_x, n_y, grid_transition):
        """
        Note: this is for single animal case
        plot the grid transition matrices. return a Figure object
        """
        # n_x corresponds to the number of columns, and n_y corresponds to the number of rows.
        grid_idx = 0
        for i in range(n_x):
            for j in range(n_y):
                # plot_idx = ij_to_plot_idx(i, j, n_x, n_y)
                # plt.subplot(n_x, n_y, plot_idx)
                ax = axn[n_y - j - 1][i]
                ax.clear()
                if i == 0 and j == 0:
                    sns.heatmap(get_np(grid_transition[grid_idx]), ax=ax, vmin=0, vmax=1, cmap="BuGn", square=True,
                                cbar_ax=cbar_ax)
                else:
                    sns.heatmap(get_np(grid_transition[grid_idx]), ax=ax, vmin=0, vmax=1, cmap="BuGn", square=True,
                                cbar=False)
                ax.tick_params(axis='both', which='both', length=0)
                grid_idx += 1

    def plot_transition(self, ax, cbar_ax, transition):
        """
        plot the heatmap for one transition matrix
        :param transition: (K, K)
        :return:
        """
        ax.clear()
        sns.heatmap(get_np(transition), ax=ax, vmin=0, vmax=1, cmap="BuGn", square=True, cbar_ax=cbar_ax)


if __name__ == "__main__":
    torch.random.manual_seed(0)
    np.random.seed(0)

    import git
    import joblib
    import matplotlib.pyplot as plt

    from project_ssms.utils import downsample
    from project_ssms.gp_observation_single import GPObservationSingle
    from project_ssms.constants import *
    from project_ssms.grid_utils import plot_realdata_quiver

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
    y_grid_gap = (ARENA_YMAX - ARENA_YMIN) / n_y
    y_grids = np.array([ARENA_YMIN + i * y_grid_gap for i in range(n_y + 1)])
    bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])
    train_rs = False
    obs = GPObservationSingle(K=K, D=D, mus_init=mus_init, x_grids=x_grids, y_grids=y_grids, bounds=bounds,
                              rs=None, train_rs=train_rs, device=device)

    L = 5
    model = HMM(K=K, D=D, init_state_distn=None, transition='stationary', observation=obs)

    ll = model.log_likelihood(data)  # -10315.3303
    print(ll)

    #z = model.most_likely_states(data)
    #plot_realdata_quiver(data, z, K=K, x_grids=x_grids, y_grids=y_grids, title="before training")

    # fitting
    num_iters = 100
    loss = model.fit(datas=data, num_iters=num_iters, lr=1e-3)  # 10178.42

    #z = model.most_likely_states(data)
    #plot_realdata_quiver(data, z, K=K, x_grids=x_grids, y_grids=y_grids, title="after {} epochs".format(num_iters))
    #plt.show()
