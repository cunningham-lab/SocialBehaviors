import sys

from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_truncated_normal_observation import ARTruncatedNormalObservation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.utils import find_permutation, random_rotation, k_step_prediction

import torch
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from tqdm import tqdm_notebook as tqdm

import time

from hips.plotting.colormaps import gradient_cmap, white_to_color_cmap
color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)


# generate synthetic data
K = 3
D = 2
T = 2
lags = 10

torch.manual_seed(0)
npr.seed(0)

bounds = np.array([[0, 20], [0,40]])
thetas = np.linspace(0, 2 * np.pi, K, endpoint=False)
mus_init = 3 * np.column_stack((np.cos(thetas), np.sin(thetas))) + 5

true_tran = LinearTransformation(K=K, D=D, lags=lags)
true_observation = ARTruncatedNormalObservation(K=K, D=D, M=0, lags=lags, transformation=true_tran, bounds=bounds, mus_init=mus_init, train_sigma=False)
true_model = HMM(K=K, D=D, M=0, observation=true_observation)

z, x = true_model.sample(T, return_np=False)
true_ll = true_model.log_likelihood(x)

print(true_ll)

print("\n # model parameters: \n", len(true_model.params))

print("\n # model trainable parameters: \n", len(true_model.trainable_params))


sample_z, sample_x = true_model.sample(T)

"""
# learning

tran = LinearTransformation(K=K, D=D, momentum_lags=1, As=As)
observation = ARTruncatedNormalObservation(K=K, D=D, M=0, momentum_lags=1, transformation=tran, bounds=bounds, mus_init=mus_init)
model = HMM(K=K, D=D, M=0, observation=true_observation)

num_iters = 5000

pbar = tqdm(total=num_iters, file=sys.stdout)

optimizer = torch.optim.Adam(model.params, lr=0.001)

losses = []
for i in np.arange(num_iters):

    optimizer.zero_grad()

    loss = model.loss(x)
    loss.backward()
    optimizer.step()

    loss = loss.detach().numpy()
    losses.append(loss)

    if i % 10 == 0:
        pbar.set_description('iter {} loss {:.2f}'.format(i, loss))
        pbar.update(10)

infer_z = model.most_likely_states(x)

perm = find_permutation(z.numpy(), infer_z, K1=K, K2=K)

model.permute(perm)
infer_z = model.most_likely_states(x)

x_predict_cond_z = k_step_prediction(model, z, x)

"""