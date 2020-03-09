"""
inference stuff. non-cythonized yet
"""
import torch
import numpy as np


def bwd_mp(tran_log_probs, seq_logprobs, fwd_obs_logprobs):
    """
    :param tran_log_probs: (T, K, K)
    :param seq_logprobs: =1 for now
    :param fwd_obs_logprobs: (L, T, K)
    :return:
    """

    L, T, K = fwd_obs_logprobs.shape


    log_betas = torch.zeros((T+1, K))
    log_beta_stars = torch.zeros((T+1, K))

    for t in range(1, T+1):
        steps_fwd = min(L, t)

        # \log beta*_t(k) = log \sum_l beta_{t+l}(k) p(x_{t+1:t+l}) p(l)
        log_beta_star = log_betas[T-t+1:T-t+1+steps_fwd] + fwd_obs_logprobs[:steps_fwd, T - t] # (steps_fwd, K) + (steps_fwd, K)

        log_beta_stars[T-t] = torch.logsumexp(log_beta_star, dim=0) # (K,)

        if t < T:
            # compute beta_t(j)
            # \log beta_t(j) = log \sum_k beta*_t(k) p(z_{t+1}=k | z_t=j)
            log_beta = log_beta_stars[T-t][None, ] + tran_log_probs[T-t-1]  # (1, K) + (K, K)
            log_beta = torch.logsumexp(log_beta, dim=1)  # (K, )
            log_betas[T-t] = log_beta

    return log_betas, log_beta_stars


# make sure to detach grad
def hsmm_viterbi(log_pi0, trans_logprobs, bwd_obs_logprobs, len_logprobs=None, L=None):
    """

    :param log_pi0: initial dist (K, )
    :param trans_logprobs: (T-1, K, K)
    :param bwd_obs_logprobs: (L, T, K), bwd_obs_logprobs[:,t] gives log prob for segments ending at t.
    :param len_logprobs: uniform for now
    :param ret_delt:
    :return:
    """

    Tm1, K, _ = trans_logprobs.shape
    T = Tm1 + 1
    if len_logprobs is None:
        assert L is not None
    else:
        L = len(len_logprobs)

    # argmax over length variable
    deltas = torch.zeros((T+1, K)) # value
    bs = torch.zeros((T, K), dtype=torch.int) # pointer

    # argmax over state variable
    delta_stars = torch.zeros((T + 1, K)) # value
    b_stars = torch.zeros((T, K), dtype=torch.int) # pointer

    delta_stars[0] = log_pi0

    for t in range(1, T+1):
        steps_back = min(L, t)

        delta_t = bwd_obs_logprobs[-steps_back:, t-1] + delta_stars[t-steps_back:t]  # (steps_back, K)
        delta_t, b_t = torch.max(delta_t, dim=0)  # (K, ), (K, )  b_t=0, length=steps_back. b_t=1, length=steps_back-1
        # if steps_back <= L, b_t=0 -> L, b_t=1 -> L-1, ...
        bs[t-1] = steps_back - b_t
        deltas[t] = delta_t

        if t < T:
            # (K, 1) + (K, K)
            delta_star_t = delta_t[:, None] + trans_logprobs[t-1]
            delta_star_t, b_star_t = torch.max(delta_star_t, dim=0)  # (K, ), (K, )
            b_stars[t-1] = b_star_t
            delta_stars[t] = delta_star_t

    seqs = recover_bp(deltas, bs, b_stars)
    return seqs


def recover_bp(deltas, bs, b_stars):
    """

    :param deltas: (T+1, K)
    :param bs: (T, K)
    :param b_stars: (T, K)
    :return: a list of lists with (start_index, end_index, label) entries / or a list of hidden states
    """
    T, K = bs.shape
    seqs = []
    hidden_states_seqs = []
    t = T
    z = torch.argmax(deltas[T])
    while True:
        d = bs[t-1, z] # int
        seq = (int(t-d+1), int(t), int(z))
        hidden_state_seq = [int(z)] * int(d)
        seqs.append(seq)
        hidden_states_seqs += hidden_state_seq
        t = t-d
        if t == 0:
            break
        z = b_stars[t-1, z]
    seqs = seqs[::-1]
    hidden_states_seqs = hidden_states_seqs[::-1]
    return seqs, hidden_states_seqs


def fwd_to_bwd(fw_logprobs):
    """
    Example:

    fw_logprobs: (for some k), T=4, L=3
    [[p1, p2, p3, p4]
    [p1:2, p2:3, p3:4, -inf]
    [p1:3, p2:4, -inf, -inf]
    [p1:4, -inf, -inf, -inf]]
    -->
    bw_logprobs:
    [[-inf, -inf, -inf, p1:4]
    [-inf, -inf, p1:3, p2:4]
    [-inf, p1:2, p2:3, p3:4]
    [p1, p2, p3, p4]]

    :param fw_logprobs: (L, T, K). First dimension: duration=1, duration=2, ..., duration=L.
    fw_logprobs[:,t,k] represents the logprobs of p(x_{t:t+d-1}|z_t=k) for d=1:L. I.e. the prob starting at t
    :return: bw_logprobs: (L, T, K). First dimension: duration=L, duration=L-1, ..., duration=1.
    bw_logprobs[:,t,k[] represents the logprobs of p(x_{t-d+1:t}|z_y=k) for d=L:1. I.e. thhe prob ending at t.
    """
    assert len(fw_logprobs.shape) == 3, fw_logprobs.shape

    L = len(fw_logprobs)
    #bw_logprobs = fw_logprobs.new().resize_as_(fw_logprobs).fill_(-float("inf"))
    bw_logprobs = fw_logprobs.new_full(fw_logprobs.shape, -float("inf"))
    bw_logprobs[L-1].copy_(fw_logprobs[0])

    for l in range(1, L):
        bw_logprobs[L-l-1, l:].copy_(fw_logprobs[l,:-l])

    return bw_logprobs


def test_case():
    # (L, T, K)
    L = 3
    T = 3
    K = 2

    # consider uniform distribution for L
    #  first dim: d=1, d=2, d=3, ..., d=L
    bws_obs_logprobs_k0 = torch.tensor([[-np.inf, -np.inf, 5], [-np.inf, 2, 4], [3, 9, 2]], dtype=torch.float)  # (L, T)
    bws_obs_logprobs_k1 = torch.tensor([[-np.inf, -np.inf, 2], [-np.inf, 3, 6], [2, 10, 3]],
                                       dtype=torch.float)  # (L, T)
    bws_obs_logprobs = torch.stack([bws_obs_logprobs_k0, bws_obs_logprobs_k1], dim=2)  # (L, T, K)

    # (T-1, K, K)
    trans_log_probs = torch.log(torch.tensor([[[0.2, 0.8], [0.8, 0.2]], [[0.1, 0.9], [0.9, 0.1]]], dtype=torch.float))

    # (K, )
    pi = torch.tensor([0.3, 0.7], dtype=torch.float)
    log_pi = torch.log(pi)

    # check viterbi
    with torch.no_grad():
        seqs, hidden_state_seqs = hsmm_viterbi(log_pi, trans_log_probs, bws_obs_logprobs, L=L)
    print(seqs)
    print(hidden_state_seqs)

    # check fwd to bwd
    # log probs starting at t
    fw_obs_logprobs = torch.stack((
        torch.tensor([[3, 9, 2], [2, 4, -np.inf], [5, -np.inf, -np.inf]], dtype=torch.float),
        torch.tensor([[2, 10, 3], [3, 6, -np.inf], [2, -np.inf, -np.inf]], dtype=torch.float)
    ), dim=2)  # (L, T, K)

    bws_obs_logprobs_2 = fwd_to_bwd(fw_obs_logprobs)
    assert torch.all(bws_obs_logprobs_2 == bws_obs_logprobs)

    # check hsmm_normalizer
    log_betas, log_beta_stars = bwd_mp(trans_log_probs, 0, fwd_obs_logprobs=fw_obs_logprobs)
    log_likelihood = torch.logsumexp(log_beta_stars[0]+log_pi, dim=0)
    print(log_likelihood)


def hsmm_normalizer(log_pi, tran_log_probs, seq_logprobs, fwd_obs_logprobs):
    """

    :param log_pi: (K, )
    :param tran_log_probs: (T-1, K, K)
    :param seq_logprobs: (K, L) or None
    :param fwd_obs_logprobs: (L, T, K)
    :return: a scalar
    """
    _, log_beta_stars = bwd_mp(tran_log_probs=tran_log_probs, seq_logprobs=seq_logprobs, fwd_obs_logprobs=fwd_obs_logprobs)
    log_likelihood = torch.logsumexp(log_beta_stars[0] + log_pi, dim=0)
    return log_likelihood


if __name__ == "__main__":
    test_case()