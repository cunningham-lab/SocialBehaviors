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


def k_step_prediction(model, model_z, data, k=0, expectation=True, sample_size=100):
    """
    Conditioned on the most likely hidden states, make the k-step prediction.
    """

    #TODO: add expectation
    data = check_and_convert_to_tensor(data)

    x_predict_arr = []
    if k == 0:
        for t in range(data.shape[0]):
            x_predict = model.observation.sample_x(model_z[t], data[:t])
            x_predict_arr.append(x_predict)
    else:
        assert k>0
        # neglects t = 0 since there is no history
        for t in range(1, data.shape[0]-k):
            zx_predict = model.sample(k, prefix=(model_z[t-1:t], data[t-1:t]))
            assert zx_predict[1].shape == (k, 4)
            x_predict = zx_predict[1][k-1]
            x_predict_arr.append(x_predict)
    x_predict_arr = np.array(x_predict_arr)
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
        raise ValueError("inputs must be an ndarray or tensor.")