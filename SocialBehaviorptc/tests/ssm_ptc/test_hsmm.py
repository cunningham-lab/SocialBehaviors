import torch
import numpy as np
import unittest

from ssm_ptc.models.hsmm import HSMM


def get_lenprobs(len_scores):
    K, L = len_scores.shape
    lplist = [len_scores.new_zeros((1, K))]
    for l in range(2, L + 1):
        lplist.append(torch.log_softmax(len_scores.narrow(1, 0, l), dim=1).t())
    return lplist


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
        log_like_l_pad = log_like_l.new_empty((l-1, K)).fill_(-float("inf"))
        log_like_l = torch.cat((log_like_l, log_like_l_pad), dim=0) # (T, K)
        assert log_like_l.shape == (T, K), log_like_l.shape
        stacked_log_likes.append(log_like_l)

    stacked_log_likes = torch.stack(stacked_log_likes, dim=0)  # (L, T, K)

    assert stacked_log_likes.shape == (max_L, T, K)
    return stacked_log_likes


class TestMarginalization(unittest.TestCase):
    def test_infc_marginalization(self):
        from ssm_ptc.message_passing.infc import hsmm_normalizer

        torch.manual_seed(0)

        T = 2
        L = 2
        K = 2

        log_pi = torch.log_softmax(torch.randn((K,), dtype=torch.float64), dim=-1)  # (K, )
        tran_logprobs = torch.log_softmax(torch.ones((K,K), dtype=torch.float64) + torch.randn((K, K), dtype=torch.float64), dim=-1)  # (K, K)
        tran_logprobs = tran_logprobs.expand((T-1, K, K))
        len_scores = torch.ones((1, L), dtype=torch.float64).expand(K, L)
        len_logprobs = get_lenprobs(len_scores)  # a list of tensors, [1xK tensor, 2xK tensor, ..., LxK tensor]
        obs_logprobs = torch.ones((T,K), dtype=torch.float64) + torch.randn((T,K), dtype=torch.float64)  # (T, K)
        fwd_obs_logprobs = stacked_fw_log_likes_helper(obs_logprobs, L)  # (L, T, K)

        # message passing version
        log_likelihood = hsmm_normalizer(log_pi, tran_logprobs, len_logprobs, fwd_obs_logprobs)

        # brute-force version
        log_likes = []
        def get_joint_likes(zs, Ls):
            t = 0
            log_likelihood = 0
            log_transition = log_pi[zs[0]]  # a scalar
            while True:
                steps_fwd = min(L, T - t)
                log_len = len_logprobs[steps_fwd - 1][Ls[t] - 1, zs[t]]
                log_seg = fwd_obs_logprobs[Ls[t] - 1, t, zs[t]]
                log_likelihood += log_seg + log_transition + log_len

                t = t + Ls[t]
                if t == T:
                    break
                if t > T:
                    raise ValueError()
                log_transition = tran_logprobs[t-1][zs[t - 1], zs[t]]
            return log_likelihood

        Ls_and_zs_list = [(torch.tensor([1,1]), torch.tensor([0,0])),
                          (torch.tensor([1,1]), torch.tensor([0,1])),
                          (torch.tensor([1,1]), torch.tensor([1,0])),
                          (torch.tensor([1,1]), torch.tensor([1,1])),
                          (torch.tensor([2,1]), torch.tensor([0,0])),
                          (torch.tensor([2,1]), torch.tensor([1,1]))
                          ]
        for Ls, zs in Ls_and_zs_list:
            log_like = get_joint_likes(zs, Ls)
            log_likes.append(log_like)
        log_likes = torch.stack(log_likes, dim=0)
        log_likelihood_bf = torch.logsumexp(log_likes, dim=-1)

        assert torch.allclose(log_likelihood, log_likelihood_bf), \
            "message passing ll = {}, brute-force ll = {}".format(log_likelihood, log_likelihood_bf)

        T = 3
        L = 2
        K = 2

        log_pi = torch.log_softmax(torch.randn((K,), dtype=torch.float64), dim=-1)  # (K, )
        tran_logprobs = torch.log_softmax(
            torch.ones((K, K), dtype=torch.float64) + torch.randn((K, K), dtype=torch.float64), dim=-1)  # (K, K)
        tran_logprobs = tran_logprobs.expand((T - 1, K, K))
        len_scores = torch.ones((1, L), dtype=torch.float64).expand(K, L)
        len_logprobs = get_lenprobs(len_scores)  # a list of tensors, [1xK tensor, 2xK tensor, ..., LxK tensor]
        obs_logprobs = torch.ones((T, K), dtype=torch.float64) + torch.randn((T, K), dtype=torch.float64)  # (T, K)
        fwd_obs_logprobs = stacked_fw_log_likes_helper(obs_logprobs, L)  # (L, T, K)

        # message passing version
        log_likelihood = hsmm_normalizer(log_pi, tran_logprobs, len_logprobs, fwd_obs_logprobs)

        # brute-force version
        log_likes = []
        Ls_and_zs_list = [(torch.tensor([1,1,1]), torch.tensor([0,0,0])), (torch.tensor([1,1,1]), torch.tensor([0,0,1])),
                          (torch.tensor([1,1,1]), torch.tensor([0,1,0])), (torch.tensor([1,1,1]), torch.tensor([0,1,1])),
                          (torch.tensor([1,1,1]), torch.tensor([1,0,0])), (torch.tensor([1,1,1]), torch.tensor([1,0,1])),
                          (torch.tensor([1,1,1]), torch.tensor([1,1,0])), (torch.tensor([1,1,1]), torch.tensor([1,1,1])),
                          (torch.tensor([2,1,1]), torch.tensor([0,0,0])), (torch.tensor([2,1,1]), torch.tensor([0,0,1])),
                          (torch.tensor([2,1,1]), torch.tensor([1,1,0])), (torch.tensor([2,1,1]), torch.tensor([1,1,1])),
                          (torch.tensor([1,2,1]), torch.tensor([0,0,0])), (torch.tensor([1,2,1]), torch.tensor([0,1,1])),
                          (torch.tensor([1,2,1]), torch.tensor([1,0,0])), (torch.tensor([1,2,1]), torch.tensor([1,1,1]))]

        for Ls, zs in Ls_and_zs_list:
            log_like = get_joint_likes(zs, Ls)
            log_likes.append(log_like)
        log_likes = torch.stack(log_likes, dim=0)
        log_likelihood_bf = torch.logsumexp(log_likes, dim=-1)
        assert torch.allclose(log_likelihood, log_likelihood_bf), \
            "message passing ll = {}, brute-force ll = {}".format(log_likelihood, log_likelihood_bf)

    def test_marginalization(self):
        from project_ssms.gp_observation_single import GPObservationSingle
        device = torch.device("cpu")

        torch.random.manual_seed(2)
        np.random.seed(1)

        data = torch.tensor([[0, 0], [10, 10]], dtype=torch.float64)

        T = 2
        D = 2
        K = 2
        L = 2

        n_x = 3
        n_y = 3

        ARENA_XMIN, ARENA_XMAX = -20, 20
        ARENA_YMIN, ARENA_YMAX = -20, 20
        mus_init = data[0] * torch.ones(K, D, dtype=torch.float64, device=device)
        x_grid_gap = (ARENA_XMAX - ARENA_XMIN) / n_x
        x_grids = np.array([ARENA_XMIN + i * x_grid_gap for i in range(n_x + 1)])
        y_grid_gap = (ARENA_YMAX - ARENA_YMIN) / n_y
        y_grids = np.array([ARENA_YMIN + i * y_grid_gap for i in range(n_y + 1)])
        bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])
        train_rs = False
        obs = GPObservationSingle(K=K, D=D, mus_init=mus_init, x_grids=x_grids, y_grids=y_grids, bounds=bounds,
                                  rs=None, train_rs=train_rs, device=device)

        model = HSMM(K=K, D=D, L=L, observation=obs)

        log_marginal_likelihood = model.log_likelihood(data) # -41.7526

        joint_likes = []
        Ls_and_zs_list = [(torch.tensor([1, 1]), torch.tensor([0, 0])),
                          (torch.tensor([1, 1]), torch.tensor([0, 1])),
                          (torch.tensor([1, 1]), torch.tensor([1, 0])),
                          (torch.tensor([1, 1]), torch.tensor([1, 1])),
                          (torch.tensor([2, 1]), torch.tensor([0, 0])),
                          (torch.tensor([2, 1]), torch.tensor([1, 1]))
                          ]

        for Ls, zs in Ls_and_zs_list:
            log_joint_likelihood = model.log_joint_likelihood(zs=zs, Ls=Ls, data=data)
            joint_likes.append(log_joint_likelihood)
        joint_likes = torch.stack(joint_likes) # (24, )
        log_marginal_likelihood2 = torch.logsumexp(joint_likes, dim=-1)

        assert torch.allclose(log_marginal_likelihood, log_marginal_likelihood2), \
            "message passing ll = {}, brute-force ll = {}".format(log_marginal_likelihood, log_marginal_likelihood2)

    def test_stacked_log_likes(self):
        p1_k1, p2_k1, p3_k1, p4_k1 = 1.5, 2.5, 3, 4
        p1_k2, p2_k2, p3_k2, p4_k2 = 1.3, 2.4, 2.5, 6.6
        log_likes = torch.tensor([[p1_k1, p1_k2], [p2_k1, p2_k2], [p3_k1, p3_k2], [p4_k1, p4_k2]])

        # L<T
        T = 4
        L = 3

        true_stacked_log_likes_k1 = torch.tensor([[p1_k1, p2_k1, p3_k1, p4_k1],
                                                  [p1_k1 + p2_k1, p2_k1 + p3_k1, p3_k1 + p4_k1, -float("inf")],
                                                  [p1_k1 + p2_k1 + p3_k1, p2_k1 + p3_k1 + p4_k1, -float("inf"),
                                                   -float("inf")]])
        true_stacked_log_likes_k2 = torch.tensor([[p1_k2, p2_k2, p3_k2, p4_k2],
                                                  [p1_k2 + p2_k2, p2_k2 + p3_k2, p3_k2 + p4_k2, -float("inf")],
                                                  [p1_k2 + p2_k2 + p3_k2, p2_k2 + p3_k2 + p4_k2, -float("inf"),
                                                   -float("inf")]])
        true_stacked_log_likes = torch.stack([true_stacked_log_likes_k1, true_stacked_log_likes_k2], dim=2)  # (L, T, K)

        s_over_L = HSMM.stacked_fw_log_likes_helper(log_likes=log_likes, L=L)
        assert torch.all(true_stacked_log_likes == s_over_L), "true = \n{}\n computed = \n{}".format(
            true_stacked_log_likes, s_over_L)

        s_over_T = HSMM.stacked_log_likes_helper_over_T(log_likes=log_likes, L=L)
        assert torch.all(true_stacked_log_likes == s_over_T), "true = \n{}\n computed = \n{}".format(
            true_stacked_log_likes, s_over_T)

        #  L >= T
        L = 5
        l4_probs = torch.tensor([[p1_k1 + p2_k1 + p3_k1 + p4_k1, p1_k2 + p2_k2 + p3_k2 + p4_k2],
                                 [-float("inf"), -float("inf")],
                                 [-float("inf"), -float("inf")],
                                 [-float("inf"), -float("inf")]])  # (T, K)
        true_stacked_log_likes = torch.cat((true_stacked_log_likes, l4_probs[None,]))  # (L, T, K)

        s_over_L = HSMM.stacked_fw_log_likes_helper(log_likes=log_likes, L=L)
        assert torch.allclose(true_stacked_log_likes, s_over_L), "true = \n{}\n computed = \n{}".format(
            true_stacked_log_likes, s_over_L)

        s_over_T = HSMM.stacked_log_likes_helper_over_T(log_likes=log_likes, L=L)
        assert torch.allclose(true_stacked_log_likes, s_over_T), "true = \n{}\n computed = \n{}".format(
            true_stacked_log_likes, s_over_T)

if __name__ == "__main__":
    #test_marginalization()
    unittest.main()