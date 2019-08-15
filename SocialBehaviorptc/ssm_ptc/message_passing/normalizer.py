import torch
from torch.autograd import Function
from ssm_ptc.message_passing.hmm import forward_pass as forward_pass_cython
from ssm_ptc.message_passing.hmm import backward_pass as backward_pass_cython
import numpy as np


class HMMNormalizerCython(Function):

    @staticmethod
    def forward(ctx, log_pi0, log_As, log_likes):
        T, K = log_likes.shape
        log_pi0, log_As, log_likes = log_pi0.detach(),\
                                     log_As.detach(),\
                                     log_likes.detach()
        # (K, ), (T, K, K), (T, K)
        alphas = np.zeros((T, K))
        Z = forward_pass_cython(log_pi0.numpy(),
                                log_As.numpy(),
                                log_likes.numpy(),
                                alphas)
        ctx.save_for_backward(torch.tensor(alphas, dtype=torch.float64),
                              log_As)
        return torch.tensor(Z, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        alphas, log_As = ctx.saved_tensors
        alphas, log_As = alphas.detach().numpy(), log_As.detach().numpy()
        T, K = alphas.shape

        d_log_pi0 = np.zeros(K)
        d_log_As = np.zeros((T - 1, K, K))
        d_log_likes = np.zeros((T, K))

        backward_pass_cython(log_As, alphas, d_log_pi0, d_log_As, d_log_likes)

        return torch.tensor(d_log_pi0, dtype=torch.float64) * grad_output, \
               torch.tensor(d_log_As, dtype=torch.float64) * grad_output, \
               torch.tensor(d_log_likes, dtype=torch.float64) * grad_output

hmmnorm_cython = HMMNormalizerCython.apply