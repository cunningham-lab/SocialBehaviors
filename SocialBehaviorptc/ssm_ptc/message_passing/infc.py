"""
inference stuff. non-cythonized yet
"""
import torch
import numpy as np

# TODO: check if -infs in tran_logprobs violates anything
def bwd_mp(tran_logprobs, len_logprobs, fwd_obs_logprobs):
    """
    :param tran_logprobs: (T-1, K, K)
    :param len_logprobs:  [(steps_fwd, K) tensor for steps_fwd in 1, ..., L], log_probs for duration variables
    :param fwd_obs_logprobs: (L, T, K)
    :return:
    """

    L, T, K = fwd_obs_logprobs.shape
    assert tran_logprobs.shape == (T-1, K, K), tran_logprobs.shape
    assert fwd_obs_logprobs.shape == (L, T, K), fwd_obs_logprobs.shape
    log_betas = tran_logprobs.new_zeros((T + 1, K))
    log_beta_stars = tran_logprobs.new_zeros((T + 1, K))

    for t in range(1, T+1): # t recorders the number of steps to the end
        steps_fwd = min(L, t)
        len_logprob = len_logprobs[steps_fwd-1]
        #len_logprob = len_logprobs[min(L-1, steps_fwd-1)]  # (steps_fwd, K)

        # \log beta*_t(k) = log \sum_l beta_{t+l}(k) p(l_{t+1}=l |k) p(x_{t+1:t+l}) p(l)
        log_beta_star = log_betas[T-t+1:T-t+1+steps_fwd] + len_logprob \
                        + fwd_obs_logprobs[:steps_fwd, T - t] # (steps_fwd, K) + (steps_fwd, K)
        log_beta_stars[T-t] = torch.logsumexp(log_beta_star, dim=0) # (K,)

        if t < T:
            # compute beta_t(j)
            # \log beta_t(j) = log \sum_k beta*_t(k) p(z_{t+1}=k | z_t=j)
            log_beta = log_beta_stars[T-t][None, ] + tran_logprobs[T - t - 1]  # (1, K) + (K, K)
            log_beta = torch.logsumexp(log_beta, dim=1)  # (K, )
            log_betas[T-t] = log_beta

    return log_betas, log_beta_stars


def hsmm_viterbi(log_pi0, trans_logprobs, bwd_obs_logprobs, len_logprobs):
    """
    # TODO: no need to flip len_logprobs for now, since it is uniform
    :param log_pi0: initial dist (K, )
    :param trans_logprobs: (T-1, K, K)
    :param bwd_obs_logprobs: (L, T, K), bwd_obs_logprobs[:,t] gives log prob for segments ending at t.
    More specifically, bws_obs_logprobs[-steps_back:, t] = p(x_{t-d+1:t}) fot d from #largest possible steps back to 1,
    :param len_logprobs: [(steps_fwd, K) tensor for steps_fwd in 1, ..., L], log_probs for duration variables
    :return:
    """

    Tm1, K, _ = trans_logprobs.shape
    T = Tm1 + 1
    L = len(len_logprobs)

    # currently len_logprobs contains tensors that are [1 step back; 2 steps back; ... L steps_back]
    # but we need to flip on the 0'th axis
    flipped_len_logprobs = []
    for l in range(len(len_logprobs)):
        llps = len_logprobs[l]
        flipped_len_logprobs.append(torch.stack([llps[-i - 1] for i in range(llps.size(0))]))

    # argmax over length variable
    deltas = torch.zeros((T+1, K)) # value
    bs = torch.zeros((T, K), dtype=torch.int) # pointer

    # argmax over state variable
    #delta_stars = torch.zeros((T + 1, K)) # value
    delta_stars = log_pi0.new_zeros((T+1, K))
    b_stars = torch.zeros((T, K), dtype=torch.int)  # pointer

    delta_stars[0] = log_pi0

    for t in range(1, T+1):
        steps_back = min(L, t)
        steps_fwd = min(L, T-t+1)

        if steps_back <= steps_fwd:
            # steps_fwd x K -> steps_back x K
            len_terms = flipped_len_logprobs[min(L-1, steps_fwd-1)][-steps_back:]
        else: # we need to pick probs from different distributions...
            len_terms = torch.stack([len_logprobs[min(L, T+1-t+jj)-1][jj]
                                     for jj in range(L-1, -1, -1)])

        delta_t = bwd_obs_logprobs[-steps_back:, t-1] + delta_stars[t-steps_back:t]  # (steps_back, K)
        #print("len_terms shape", len_terms.shape)
        #print("delta_t terms shape", delta_t.shape)
        delta_t = delta_t + len_terms
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

    seqs, hidden_state_seqs = recover_bp(deltas, bs, b_stars)
    return seqs, hidden_state_seqs


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
    return seqs, np.array(hidden_states_seqs)


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
    [p1,   p2,   p3,   p4]]

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
    print(log_likelihood)  # (14.4752)


def hsmm_normalizer(log_pi, tran_logprobs, len_logprobs, fwd_obs_logprobs):
    """

    :param log_pi: (K, )
    :param tran_logprobs: (T-1, K, K)
    :param len_logprobs: (K, L) or None
    :param fwd_obs_logprobs: (L, T, K)
    :return: a scalar
    """
    _, log_beta_stars = bwd_mp(tran_logprobs=tran_logprobs, len_logprobs=len_logprobs, fwd_obs_logprobs=fwd_obs_logprobs)
    log_likelihood = torch.logsumexp(log_beta_stars[0] + log_pi, dim=0)
    return log_likelihood


if __name__ == "__main__":
    test_case()