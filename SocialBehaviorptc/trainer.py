from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_gaussian_observation import ARGaussianObservation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.utils import find_permutation, random_rotation

import torch
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    pass


torch.manual_seed(0)

K = 3
D = 2
T = 100

As = [random_rotation(D) for _ in range(K)]
true_tran = LinearTransformation(K=K, d_in=D, D=D, As=As)
true_observation = ARGaussianObservation(K=K, D=D, M=0, transformation=true_tran)
true_model = HMM(K=K, D=D, M=0, observation=true_observation)

z, data = true_model.sample(T)

#fake_data = torch.tensor([[0.4294], [-0.0689]], dtype=torch.float64)
#log_prob = model.log_likelihood(fake_data)

# now fit the data

tran = LinearTransformation(K=K, d_in=D, D=D)
observation = ARGaussianObservation(K=K, D=D, M=0, transformation=tran)
model = HMM(K=K, D=D, M=0, observation=observation)


num_iters = 20


optimizer = torch.optim.Adam(model.params, lr=0.001)

losses = []
for i in np.arange(9000):

    optimizer.zero_grad()

    print(i)
    loss = model.loss(data)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(loss)
    #print(data)
    losses.append(loss.detach().numpy())


infer_z = model.most_likely_states(data)

perm = find_permutation(z.numpy(), infer_z, K1=K, K2=K)

model.permute(perm)
hmm_z = model.most_likely_states(data)

true_loss = true_model.loss(data)
plt.plot(losses)
plt.plot(np.arange(num_iters), [true_loss]*num_iters)
plt.show()


