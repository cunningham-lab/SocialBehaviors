import torch
import numpy as np

from project_ssms.coupled_momentum_observation import CoupledMomentumObservation
from project_ssms.feature_funcs import feature_func_single

from ssm_ptc.models.hmm import HMM

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

data = torch.tensor(traj0, dtype=torch.float64)


arena_xmax = 320
arena_ymax = 380


####################### Models ##########################

bounds = np.array([[10, arena_xmax + 10], [-10, arena_ymax + 10], [10, arena_xmax + 10],
                   [-10, arena_ymax + 10]])

K = 4
D = 4
lags = 20
Df = 10
T = 36000


observation = CoupledMomentumObservation(K=K, D=D, M=0, lags=lags, Df=Df, feature_func=feature_func_single)

model = HMM(K=K, D=D, M=0, observation=observation)

# precompute features

momentum_vecs = model.observation.transformation._compute_momentum_vecs(data[:-1])
features = model.observation.transformation._compute_features(data[:-1])

out = model.log_likelihood(data, momentum_vecs=momentum_vecs, features=features)


##################### training ############################

num_iters = 10
losses, opt = model.fit(data, num_iters=num_iters, lr=0.001, momentum_vecs=momentum_vecs, features=features)


##################### sampling ############################

sample_z, sample_x = model.sample(30)






