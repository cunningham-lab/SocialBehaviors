import torch
import numpy as np

from project_ssms.momentum_feature_observation import MomentumFeatureObservation, \
    MomentumFeatureTransformation
from project_ssms.momentum_utils import filter_traj_by_speed
from project_ssms.feature_funcs import feature_func_single
from project_ssms.utils import k_step_prediction_for_momentum_feature_model

from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

import joblib


torch.manual_seed(0)
np.random.seed(0)


######################## Datasets ########################

datasets_processed = \
    joblib.load('/Users/leah/Columbia/courses/19summer/SocialBehavior/tracedata/all_data_3_1')  # a list of length 30, each is a social_dataset

rendered_data = []
for dataset in datasets_processed:
    session_data = dataset.render_trajectories([3,8])  # list of length 2, each item is an array (T, 2). T = 36000
    rendered_data.append(np.concatenate((session_data),axis = 1)) # each item is an array (T, 4)
trajectories = np.concatenate(rendered_data,axis=0)  # (T*30, 4)

traj0 = rendered_data[0]

f_traj = filter_traj_by_speed(traj0, q1=0.99, q2=0.99)

data = torch.tensor(f_traj, dtype=torch.float64)


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
