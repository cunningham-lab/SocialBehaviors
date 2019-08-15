import torch
from torch.autograd import Function
from .hmm import forward_pass as forward_pass_cython
from .hmm import backward_pass as backward_pass_cython
import numpy as np

class HMMNormalizerCython(Function):

    @staticmethod
    def forward(ctx, log_pi0, log_As, log_likes):
        T, K = log_likes.shape
        log_pi0, log_As, log_likes = log_pi0.detach(),\
                                     log_As.detach(),\
                                     log_likes.detach()
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

        return torch.tensor(d_log_pi0 * grad_output, dtype=torch.float64), \
               torch.tensor(d_log_As * grad_output, dtype=torch.float64), \
               torch.tensor(d_log_likes * grad_output, dtype=torch.float64)

hmmnorm_cython = HMMNormalizerCython.apply