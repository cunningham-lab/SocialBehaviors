"""
compute_state_overlap, find_permutation, random_rotation are copied from https://github.com/slinderman/ssm
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

import time, sys


def compute_state_overlap(z1, z2, K1=None, K2=None):
    #assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape
    assert K1 <= K2, "Can only find permutation from more states to fewer"

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm


def random_rotation(n, theta=None):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n, n))
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


def k_step_prediction(model, model_z, data, k=0):
    """
    Conditioned on the most likely hidden states, make the k-step prediction.
    """

    data = check_and_convert_to_tensor(data)
    T, D = data.shape

    x_predict_arr = []
    if k == 0:
        for t in range(T):
            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True)
            x_predict_arr.append(x_predict)
    else:
        assert k>0
        # neglects t = 0 since there is no history

        if T <= k:
            raise ValueError("Please input k such that k < {}.".format(T))

        for t in range(1, T-k+1):
            # TODO: fix k-step prediction sample size
            zx_predict = model.sample(k, prefix=(model_z[:t], data[:t]), return_np=True)
            assert zx_predict[1].shape == (k, D)
            x_predict = zx_predict[1][-1]
            x_predict_arr.append(x_predict)
    x_predict_arr = np.array(x_predict_arr)

    assert x_predict_arr.shape == (T-k, D)
    return x_predict_arr


def check_and_convert_to_tensor(inputs, dtype=torch.float64):
    """
    check if inputs type is either ndarray or tensor (requires_grad=False)
    :param inputs:
    :param dtype: the torch.dtype that inputs should be converted to
    :return: converted tensor
    """
    if isinstance(inputs, np.ndarray):
        return torch.tensor(inputs, dtype=dtype)
    elif isinstance(inputs, torch.Tensor):
        return inputs
    else:
        raise ValueError("Inputs must be an ndarray or tensor.")


def get_np(input):
    if isinstance(input, np.ndarray):
        return np
    elif isinstance(input, torch.Tensor):
        if input.requires_grad:
           return input.detach().numpy()
        else:
            return input.numpy()
    raise ValueError("Inputs must be an ndarray or tensor.")


def set_param(param, value):
    return torch.tensor(get_np(value), dtype=param.dtype, requires_grad=param.requires_grad)


def ensure_args_are_lists_of_tensors(f):
    def wrapper(self, datas, inputs=None, **kwargs):
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas

        for i, data in enumerate(datas):
            datas[i] = check_and_convert_to_tensor(data)

        batch_size = len(datas)

        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)

        _default_inputs = [None for _ in range(batch_size)]

        if inputs is None:
            inputs = _default_inputs
        elif inputs != _default_inputs:
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            for i, input in enumerate(inputs):
                inputs[i] = check_and_convert_to_tensor(input)

            assert len(inputs) == batch_size

        return f(self, datas, inputs, **kwargs)

    return wrapper
