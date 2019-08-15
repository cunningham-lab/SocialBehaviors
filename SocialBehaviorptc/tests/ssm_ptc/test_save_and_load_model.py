import torch
import numpy as np

from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_gaussian_observation import ARGaussianObservation
from ssm_ptc.transformations.linear import LinearTransformation


import joblib

torch.manual_seed(0)
np.random.seed(0)


K = 3
D = 2
lags= 10

# AR Gaussian

trans1 = LinearTransformation(K=K, D=D, lags=lags)
obs1 = ARGaussianObservation(K=K, D=D, transformation=trans1, train_sigma=False)
model = HMM(K=K,D=D, observation=obs1)

filename = "test_save_model"
joblib.dump(model, filename)


model_recovered = joblib.load(filename)

for p1, p2 in zip(model.params_unpack, model_recovered.params_unpack):
    assert torch.all(torch.eq(p1, p2))

