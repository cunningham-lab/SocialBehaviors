import torch
import numpy as np

from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_gaussian_observation import ARGaussianObservation
from ssm_ptc.observations.ar_logit_normal_observation import ARLogitNormalObservation
from ssm_ptc.observations.ar_truncated_normal_observation import ARTruncatedNormalObservation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.transformations.constrained_linear import ConstrainedLinearTransformation
from ssm_ptc.utils import k_step_prediction

torch.manual_seed(0)
np.random.seed(0)


K = 3
D = 2
lags = 10

# AR Gaussian

trans1 = LinearTransformation(K=K, D=D, lags=lags)
obs1 = ARGaussianObservation(K=K, D=D, transformation=trans1, train_sigma=False)
model1 = HMM(K=K,D=D, observation=obs1)

model2 = HMM(K=K, D=D, observation_kwargs={"lags": lags})

#print(model1.params == model2.params)

model2.params = model1.params

for p1, p2 in zip(model1.params_unpack, model2.params_unpack):
    assert torch.all(torch.eq(p1, p2))


# AR LogitNormal

bounds = np.array([[0,2], [0,4]])

model1 = HMM(K=K,D=D, observation='logitnormal', observation_kwargs={"lags": lags, "bounds": bounds})

model2 = HMM(K=K, D=D, observation='logitnormal', observation_kwargs={"lags": lags, "bounds": bounds})

#print(model1.params == model2.params)

model2.params = model1.params

for p1, p2 in zip(model1.params_unpack, model2.params_unpack):
    assert torch.all(torch.eq(p1, p2))

# AR TruncatedNormal


model1 = HMM(K=K,D=D, observation='truncatednormal', observation_kwargs={"lags": lags, "bounds": bounds})

model2 = HMM(K=K, D=D, observation='truncatednormal', observation_kwargs={"lags": lags, "bounds": bounds})

model2.params = model1.params

for p1, p2 in zip(model1.params_unpack, model2.params_unpack):
    assert torch.all(torch.eq(p1, p2))
