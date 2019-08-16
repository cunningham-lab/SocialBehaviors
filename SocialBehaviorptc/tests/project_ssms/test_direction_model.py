import torch
import numpy as np

from project_ssms.coupled_transformations.direction_observation import DirectionObservation, DirectionTransformation
from project_ssms.momentum_utils import filter_traj_by_speed
from project_ssms.feature_funcs import feature_vec_func

from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

import joblib
import git


repo = git.Repo('.', search_parent_directories=True) # SocialBehaviorectories=True)
repo_dir = repo.working_tree_dir  # SocialBehavior

torch.manual_seed(0)
np.random.seed(0)


# TODO: write k-step-prediction
# TODO: test datas (training in batches)
# TODO: test model


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

momentum_lags = 30
momentum_weights = np.arange(0.55, 2.05, 0.05)

K = 4
D = 4

Df = 5
T = data.shape[0]


observation = DirectionObservation(K=K, D=D, Df=Df, bounds=bounds, momentum_lags=momentum_lags,
                                   momentum_weights=momentum_weights, feature_vec_func=feature_vec_func)

model = HMM(K=K, D=D, M=0, observation=observation)


# precompute features

momentum_vecs = DirectionTransformation._compute_momentum_vecs(data[:-1], lags=momentum_lags)
features = DirectionTransformation._compute_features(feature_vec_func=feature_vec_func, inputs=data[:-1])


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
#x_predict = k_step_prediction_for_direction_model(model, z, data, momentum_vecs=momentum_vecs, features=features)

x_predict = k_step_prediction(model, z, data, 10)
# TODO: need to revise the k-step prediction, specifically the way to calculate the momentum
