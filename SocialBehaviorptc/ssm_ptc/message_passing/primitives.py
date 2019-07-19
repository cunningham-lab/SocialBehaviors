import numpy as np


# numpy implementation
def viterbi(log_pi0, log_Ps, ll):
    """
    Find the most likely state sequence

    This is modified from pyhsmm.internals.hmm_states
    by Matthew Johnson.
    """
    T, K = ll.shape

    # Check if the transition matrices are stationary or
    # time-varying (hetero)
    hetero = (log_Ps.shape[0] == T-1)
    if not hetero:
        assert log_Ps.shape[0] == 1

    # Pass max-sum messages backward
    scores = np.zeros_like(ll)
    args = np.zeros_like(ll, dtype=int)
    for t in range(T-2,-1,-1):
        vals = log_Ps[t * hetero] + scores[t+1] + ll[t+1]
        args[t+1] = vals.argmax(axis=1)
        scores[t] = vals.max(axis=1)

    # Now maximize forwards
    z = np.zeros(T, dtype=int)
    z[0] = (scores[0] + log_pi0 + ll[0]).argmax()
    for t in range(1, T):
        z[t] = args[t, z[t-1]]

    return z
