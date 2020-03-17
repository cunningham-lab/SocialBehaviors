import torch
import numpy as np

from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.truncated_normal_observation import TruncatedNormalObservation
from ssm_ptc.utils import k_step_prediction

import joblib
import git


repo = git.Repo('.', search_parent_directories=True) # SocialBehaviorectories=True)
repo_dir = repo.working_tree_dir  # SocialBehavior

torch.manual_seed(0)
np.random.seed(0)

# fake data
T = 5

data = np.array([[1.0, 1.0, 1.0, 6.0], [3.0, 6.0, 8.0, 6.0],
                 [4.0, 7.0, 8.0, 5.0], [6.0, 7.0, 5.0, 6.0], [8.0, 2.0, 6.0, 1.0]])
data = torch.tensor(data, dtype=torch.float64)

bounds = np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 10.0], [0.0, 8.0]])

# model

K = 3
D = 4
M = 0


obs = TruncatedNormalObservation(K=K, D=D, M=M, bounds=bounds)

# model
model = HMM(K=K, D=D, M=M, observation=obs)

# log like
log_prob = model.log_probability(data)
print("log probability = ", log_prob)

# training
print("start training...")
num_iters = 10
losses, opt = model.fit(data, num_iters=num_iters, lr=0.001)

# sampling
samplt_T = T
print("start sampling")
sample_z, sample_x = model.sample(samplt_T)

print("start sampling based on with_noise")
sample_z2, sample_x2 = model.sample(T, with_noise=True)


# inference
print("inferiring most likely states...")
z = model.most_likely_states(data)


print("0 step prediction")
x_predict = k_step_prediction(model, z, data)

print("k step prediction")
x_predict_10 = k_step_prediction(model, z, data, 2)
