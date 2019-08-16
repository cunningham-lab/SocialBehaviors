import torch
import numpy as np

from project_ssms.coupled_transformations.momentum_feature_observation import MomentumFeatureObservation, \
    MomentumFeatureTransformation
from project_ssms.momentum_utils import filter_traj_by_speed
from project_ssms.feature_funcs import feature_func_single
from project_ssms.utils import k_step_prediction_for_momentum_feature_model

from ssm_ptc.models.hmm import HMM

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
momentum_lags = 50
Df = 10
T = 36000


observation = MomentumFeatureObservation(K=K, D=D, M=0, bounds=bounds, momentum_lags=momentum_lags,
                                         Df=Df, feature_funcs=feature_func_single)

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

momentum_vecs = MomentumFeatureTransformation._compute_momentum_vecs(data[:-1], lags=momentum_lags)
features = MomentumFeatureTransformation._compute_features(feature_funcs=feature_func_single, inputs=data[:-1])


out = model.log_likelihood(data, momentum_vecs=momentum_vecs, features=features)
print(out)


##################### training ############################

num_iters = 10
losses, opt = model.fit(data, num_iters=num_iters, lr=0.001, momentum_vecs=momentum_vecs, features=features)


##################### sampling ############################
print("start sampling")
sample_z, sample_x = model.sample(30)

#################### inference ###########################
print("inferiring most likely states...")
z = model.most_likely_states(data, momentum_vecs=momentum_vecs, features=features)

print("k step prediction")
x_predict = k_step_prediction_for_momentum_feature_model(model, z, data,
                                                         momentum_vecs=momentum_vecs, features=features)
#x_predict = k_step_prediction(model, z, data, 10)
# TODO: need to revise the k-step prediction, specifically the way to calculate the momentum
