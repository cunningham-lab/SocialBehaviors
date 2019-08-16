import torch
import numpy as np

from project_ssms.coupled_transformations.momentum_interaction_observation2 import MomentumInteractionObservation, \
    MomentumInteractionTransformation
from project_ssms.momentum_utils import filter_traj_by_speed
from project_ssms.feature_funcs import feature_func_single
from project_ssms.utils import k_step_prediction_for_momentum_interaction_model

from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

import joblib
import git


repo = git.Repo('.', search_parent_directories=True) # SocialBehaviorectories=True)
repo_dir = repo.working_tree_dir  # SocialBehavior


torch.manual_seed(0)
np.random.seed(0)


######################## Datasets ########################

data_dir = repo_dir +'/SocialBehaviorptc/data/trajs_all'
trajs = joblib.load(data_dir)

traj0 = trajs[36000*0:36000*1]
f_traj = filter_traj_by_speed(traj0, q1=0.99, q2=0.99)

data = torch.tensor(f_traj, dtype=torch.float64)
data = data[:100]



arena_xmin = 10
arena_xmax = 320

arena_ymin = -10
arena_ymax = 390


####################### Models ##########################

bounds = np.array([[arena_xmin - 5, arena_xmax + 5], [arena_ymin - 5, arena_ymax + 5],
                   [arena_xmin - 5, arena_xmax + 5], [arena_ymin - 5, arena_ymax + 5]])

max_v = np.array([6.0, 6.0, 6.0, 6.0])


K = 4
D = 4

momentum_lags = 30
momentum_weights = np.arange(0.55, 2.05, 0.05)

T = 36000


observation = MomentumInteractionObservation(K=K, D=D, bounds=bounds,
                                             momentum_lags=momentum_lags, momentum_weights=momentum_weights,
                                             max_v=max_v)

model = HMM(K=K, D=D, M=0, observation=observation)

"""
##################### test params ########################

obs2 = CoupledMomentumObservation(K=K, D=D, M=0, momentum_lags=momentum_lags, Df=Df, feature_func=feature_func_single, bounds=bounds)
model2 = HMM(K=K, D=D, M=0, observation=obs2)

model2.params = model.params
for p1, p2 in zip(model.params_unpack, model2.params_unpack):
    assert torch.all(torch.eq(p1, p2))

"""

# precompute features

momentum_vecs = MomentumInteractionTransformation._compute_momentum_vecs(data[:-1], lags=momentum_lags)
interaction_vecs = MomentumInteractionTransformation._compute_direction_vecs(data[:-1])


out = model.log_likelihood(data, momentum_vecs=momentum_vecs, interaction_vecs=interaction_vecs)
print(out)


##################### training ############################

num_iters = 10
losses, opt = model.fit(data, num_iters=num_iters, lr=0.001, momentum_vecs=momentum_vecs, interaction_vecs=interaction_vecs)


##################### sampling ############################
print("start sampling")
sample_z, sample_x = model.sample(30)

#################### inference ###########################
print("inferiring most likely states...")
z = model.most_likely_states(data, momentum_vecs=momentum_vecs, interaction_vecs=interaction_vecs)

print("k step prediction")
x_predict = k_step_prediction_for_momentum_interaction_model(model, z, data,
                                                         momentum_vecs=momentum_vecs, interaction_vecs=interaction_vecs)
print("k step prediction without precomputed features.")
x_predict_2 = k_step_prediction(model, z, data, 10)

