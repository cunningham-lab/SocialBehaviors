import torch
import torch.nn as nn
import numpy as np

from ssm_ptc.models.hsmm import HSMM


class VariationalHSMM(HSMM):

    def __init__(self, K, D, L, M=0, init_state_distn=None,
                 transition="stationary", transition_kwargs=None,
                 observation="gaussian", observation_kwargs=None,
                 device=torch.device('cpu')):

        super(VariationalHSMM, self).__init__(K, D, L, M,
                                              init_state_distn, transition, transition_kwargs,
                                              observation, observation_kwargs,
                                              device)

        self.encoder = Encoder()

    def elbo(self, data):

        q = self.encode(data)  # returns a distribution
        zs, Ls = q.rsample()  # should be gumbel-softmax distribution

        # evaluate joint log_likelihood
        log_likes = self.log_joint_likelihood(zs, Ls, data)

        # TODO: whether to use MC estimatees, or exact entropy?
        entropy = q.entropy()

        elbo = log_likes + entropy
        return elbo

    def log_joint_likelihood(self, zs, Ls, data):
        """
        # TODO: add cache
        :param zs: (T, )
        :param Ls: (T, ) list of integers
        :param data: (T, D)
        :return: a scalar
        """
        T, D = data.shape
        assert zs.shape == Ls.shape == (T, ), \
            "zs.shape = {}, Ls.shape = {}. data length = {}".format(zs.shape, Ls.shape, T)
        assert D == self.D, "D = {}, model.D = {}".format(D, self.D)

        t = 0
        log_likelihood = 0
        x_pre = None
        while True:
            print("interval ", [t, t+Ls[t]])
            log_seg = self.observation.log_prob_condition_on_z(data[t:t+Ls[t]], zs[t], x_pre=x_pre)
            if t+Ls[t] == T:
                break
            if t+Ls[t] > T:
                raise ValueError("Ls does not match data length")

            log_likelihood += log_seg
            t = t+Ls[t]
            x_pre = data[t-1:t]
        return log_likelihood

    def encode(self, data):
        # TODO: should be a Gumbel-Softmax distribution
        pass


class Encoder(nn.Module):
    """
    Input observation, output the distribution of Ls and zs
    """
    def __init__(self):
        super(Encoder, self).__init__()

        # a bi-directional LSTM for z

        # a bi-direction LSTM fo L


if __name__ == "__main__":
    from project_ssms.constants import *
    from project_ssms.gp_observation_single import GPObservationSingle
    device = torch.device("cpu")

    data = torch.tensor([[0, 0], [5,5], [10,10]], dtype=torch.float64)
    zs = torch.tensor([0, 1, 1], dtype=torch.int)
    Ls = torch.tensor([1, 2, 1], dtype=torch.int)

    T = 3
    D = 2
    K = 2

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

    model = VariationalHSMM(K=K, D=D, L=3, observation=obs)

    log_mariginal_likelihood = model.log_likelihood(data)
    print("log marginal likelihood", log_mariginal_likelihood)

    log_joint_likelihood = model.log_joint_likelihood(zs=zs, Ls=Ls, data=data)
    print("log joint likelihood", log_joint_likelihood)








