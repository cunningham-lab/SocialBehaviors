from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_gaussian_observation import ARGaussianObservation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.utils import find_permutation, random_rotation, k_step_prediction

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
import sys
import time

# Generate synthetic data

torch.manual_seed(0)

K = 3
D = 2
T = 100

"""
As = [random_rotation(D) for _ in range(K)]
true_tran = LinearTransformation(K=K, momentum_lags=1, D=D, As=As)
true_observation = ARGaussianObservation(K=K, D=D, M=0, with_noise=true_tran)
true_model = HMM(K=K, D=D, M=0, observation=true_observation)

z, data = true_model.sample(T, return_np=False)

# Define a model to fit the data

tran = LinearTransformation(K=K, momentum_lags=1, D=D)
observation = ARGaussianObservation(K=K, D=D, M=0, momentum_lags=1, with_noise=tran)
model = HMM(K=K, D=D, M=0, observation=observation)

# Model fitting

num_iters = 100

pbar = tqdm(total=num_iters, file=sys.stdout)

optimizer = torch.optim.Adam(model.params, lr=0.0001)

losses = []
for i in np.arange(num_iters):

    optimizer.zero_grad()

    loss = model.loss(data)
    loss.backward(retain_graph=True)
    optimizer.step()

    loss = loss.detach().numpy()
    losses.append(loss)

    if i % 10 == 0:
        pbar.set_description('iter {} loss {:.2f}'.format(i, loss))
        pbar.update(10)

x_reconstruct = model.sample_condition_on_zs(z, data[0])

"""

# test momentum_lags

true_observation = ARGaussianObservation(K=K, D=D, M=0, lags=5, transformation='linear')
true_model = HMM(K=K, D=D, M=0, observation=true_observation)

z, data = true_model.sample(T, return_np=True)


# fit to a model
observation = ARGaussianObservation(K=K, D=D, M=0, lags=5, transformation='linear')
model = HMM(K=K, D=D, M=0, observation=observation)

num_iters = 10000

pbar = tqdm(total=num_iters, file=sys.stdout)

optimizer = torch.optim.Adam(model.params, lr=0.0001)

losses = []
for i in np.arange(num_iters):

    optimizer.zero_grad()

    loss = model.loss(data)
    loss.backward(retain_graph=True)
    optimizer.step()

    loss = loss.detach().numpy()
    losses.append(loss)

    if i % 10 == 0:
        pbar.set_description('iter {} loss {:.2f}'.format(i, loss))
        pbar.update(10)