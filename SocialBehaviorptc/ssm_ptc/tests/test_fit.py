import torch
import numpy as np

from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_gaussian_observation import ARGaussianObservation
from ssm_ptc.observations.ar_logit_normal_observation import ARLogitNormalObservation
from ssm_ptc.observations.ar_truncated_normal_observation import ARTruncatedNormalObservation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.transformations.constrained_linear import ConstrainedLinearTransformation
from ssm_ptc.utils import k_step_prediction

import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

# test fitting

K = 3
D = 2
lags= 5

trans1 = LinearTransformation(K=K, D=D, lags=lags)
obs1 = ARGaussianObservation(K=K, D=D, transformation=trans1)
model1 = HMM(K=K,D=D, observation=obs1)

T = 100
sample_z, sample_x = model1.sample(T)

model2 = HMM(K=K, D=D, observation='gaussian', lags=lags)

lls, opt = model2.fit(sample_x, num_iters=2000, lr=0.001)

z_infer = model2.most_likely_states(sample_x)

x_predict = k_step_prediction(sample_x, sample_x, z_infer)

plt.figure()
plt.plot(x_predict[:,0], label='prediction')
plt.plot(sample_x[:,0].numpy(), label='truth')
plt.show()