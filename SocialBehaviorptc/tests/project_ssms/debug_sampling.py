import torch
import numpy as np

from project_ssms.coupled_transformations.grid_transformation import GridTransformation
from project_ssms.feature_funcs import feature_direction_vec
from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation

from ssm_ptc.models.hmm import HMM

import joblib


arena_xmin = 10
arena_xmax = 320
arena_ymin = -10
arena_ymax = 390

CORNERS = torch.tensor([[10, -10], [10, 390], [320, -10], [320, 390]], dtype=torch.float64)

# make 3 by 3 grid world
x_grid_gap = (arena_xmax - arena_xmin) / 3
y_grid_gap = (arena_ymax - arena_ymin) / 3

x_grids = [arena_xmin + i * x_grid_gap for i in range(4)]
y_grids = [arena_ymin + i * y_grid_gap for i in range(4)]


torch.manual_seed(0)
np.random.seed(0)

D = 4
K = 2
M = 0

Df = 4

f_corner_vec_func = lambda s: feature_direction_vec(s, CORNERS)

bounds = np.array([[arena_xmin, arena_xmax], [arena_ymin, arena_ymax],
                   [arena_xmin, arena_xmax], [arena_ymin, arena_ymax]])


tran = GridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, unit_transformation="direction",
                          Df=Df, feature_vec_func=f_corner_vec_func, acc_factor=10)

obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=1, bounds=bounds, transformation=tran)

# model
model = HMM(K=K, D=D, M=M, observation=obs)


#model.observation.mus_init = data[0] * torch.ones(K, D, dtype=torch.float64)

params = joblib.load("/Users/leah/Columbia/courses/19summer/SocialBehavior/SocialBehaviorptc/project_notebooks/gridmodel/model_k2")

model.params = params

obs.log_sigmas = torch.log(torch.ones((2,4), dtype=torch.float64)*0.01)

center_z = torch.tensor([0], dtype=torch.int)
center_x = torch.tensor([[150, 190, 200, 200]], dtype=torch.float64)

sample_z_center, sample_x_center = model.sample(20, prefix=(center_z, center_x))

diff = np.diff(sample_x_center, axis=0)

print("diff in dim 0")
print(diff[:,0])

print("\ndiff in dim 1")
print(diff[:,1])