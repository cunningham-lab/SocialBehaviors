from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_truncated_normal_observation import ARTruncatedNormalObservation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.utils import find_permutation, random_rotation, k_step_prediction

import torch
import numpy as np
import numpy.random as npr


torch.manual_seed(0)
npr.seed(0)



K = 3
D = 4
T = 10

data = torch.randn(T, D, dtype=torch.float64)

lags = 1

bounds = np.array([[-2 ,2], [0, 1], [-2, 2], [0,1]])

As = np.array([np.column_stack([np.identity(D), np.zeros((D, (lags-1) * D))]) for _ in range(K)])

torch.manual_seed(0)
np.random.seed(0)

tran = LinearTransformation(K=K, D=D, lags=lags, As=As)
observation = ARTruncatedNormalObservation(K=K, D=D, M=0, transformation=tran, bounds=bounds)

model = HMM(K=K, D=D, M=0, observation=observation)

lls = model.log_likelihood(data)
print(lls)


#losses_1, optimizer_1 = model_1.fit(data_1, method='adam', num_iters=2000, lr=0.001)

z_1 = model.most_likely_states(data)

x_predict_arr_lag1 = k_step_prediction(model, z_1, data)