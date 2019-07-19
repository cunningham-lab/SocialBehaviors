from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_sigmoid_normal_observation import ARSigmoidNormalObservation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.utils import find_permutation, random_rotation, k_step_prediction

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
import sys

torch.manual_seed(0)

K = 3
D = 2
T = 100

As = [random_rotation(D) for _ in range(K)]
true_tran = LinearTransformation(K=K, d_in=D, D=D, As=As)

bounds = np.array([[0, 20], [-5, 25]])
true_observation = ARSigmoidNormalObservation(K=K, D=D, M=0, transformation=true_tran, bounds=bounds)

true_model = HMM(K=K, D=D, M=0, observation=true_observation)

z, data = true_model.sample(T, return_np=False)


# Define a model to fit the data

tran = LinearTransformation(K=K, d_in=D, D=D)
observation = ARSigmoidNormalObservation(K=K, D=D, M=0, transformation=tran, bounds=bounds)
model = HMM(K=K, D=D, M=0, observation=observation)

# Model fitting

num_iters = 10 #9000

pbar = tqdm(total=num_iters, file=sys.stdout)

optimizer = torch.optim.Adam(model.params, lr=0.001)

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

# check reconstruction
x_reconstruct = model.sample_condition_on_zs(z, data[0])

# infer the latent states
infer_z = model.most_likely_states(data)

perm = find_permutation(z.numpy(), infer_z, K1=K, K2=K)

model.permute(perm)
hmm_z = model.most_likely_states(data)

# check prediction
x_predict_cond_z = k_step_prediction(model, z, data)