from project_ssms.gp_observation_single import GPObservationSingle
from project_ssms.gp_observation_single_original import GPObservationSingle as GPObservationSingleOriginal
from project_ssms.constants import *
from project_ssms.utils import downsample
from ssm_ptc.models.hmm import HMM

import git
import torch
import numpy as np
import joblib

torch_seed = 0
np_seed = 0
animal = 'virgin'
video_clip_start = 0
video_clip_end = 0.001
downsample_n = 8
held_out_proportion = 0.3
x_grids = None
y_grids = None
n_x = 1
n_y = 1
K = 1
rs = None
vs = None
train_rs = True
transition = 'grid'

repo = git.Repo('.', search_parent_directories=True)  # SocialBehaviorectories=True)
repo_dir = repo.working_tree_dir  # SocialBehavior

torch.manual_seed(torch_seed)
np.random.seed(np_seed)

########################## data ########################
data_dir = repo_dir + '/SocialBehaviorptc/data/trajs_all'
trajs = joblib.load(data_dir)

if animal == 'virgin':
    trajs = trajs[:,0:2]
elif animal == 'mother':
    trajs = trajs[:,2:4]

traj = trajs[int(36000*video_clip_start):int(36000*video_clip_end)]
traj = downsample(traj, downsample_n)

device = torch.device('cpu')
data = torch.tensor(traj, dtype=torch.float64, device=device)
assert 0 <= held_out_proportion <= 0.4, \
    "held_out-portion should be between 0 and 0.4 (inclusive), but is {}".format(held_out_proportion)
T = data.shape[0]
breakpoint = int(T*(1-held_out_proportion))
training_data = data[:breakpoint]
valid_data = data[breakpoint:]


# model
D = data.shape[1]
assert D == 4 or D == 2, D
M = 0

if D == 4:
    bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX],
                       [ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])
else:
    bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])

if x_grids is None:
    x_grid_gap = (ARENA_XMAX - ARENA_XMIN) / n_x
    x_grids = np.array([ARENA_XMIN + i * x_grid_gap for i in range(n_x + 1)])
else:
    x_grids = np.array([float(x) for x in x_grids.split(",")])
    n_x = len(x_grids) - 1

if y_grids is None:
    y_grid_gap = (ARENA_YMAX - ARENA_YMIN) / n_y
    y_grids = np.array([ARENA_YMIN + i * y_grid_gap for i in range(n_y + 1)])
else:
    y_grids = np.array([float(x) for x in y_grids.split(",")])
    n_y = len(y_grids) - 1

mus_init = training_data[0] * torch.ones(K, D, dtype=torch.float64, device=device)
if animal == 'both':
    obs = GPObservation(K=K, D=D, mus_init=mus_init, x_grids=x_grids, y_grids=y_grids, bounds=bounds,
                        rs=rs, train_rs=train_rs, train_vs=train_vs, device=device)
else:
    obs = GPObservationSingle(K=K, D=D, mus_init=mus_init, x_grids=x_grids, y_grids=y_grids, bounds=bounds,
                              rs=rs, train_rs=train_rs, device=device)

if transition == 'sticky':
    transition_kwargs = dict(alpha=sticky_alpha, kappa=sticky_kappa)
elif transition == 'grid':
    transition_kwargs = dict(x_grids=x_grids, y_grids=y_grids)
else:
    transition_kwargs = None

model = HMM(K=K, D=D, M=M, transition=transition, observation=obs, transition_kwargs=transition_kwargs,
            device=device)
# TODO set parameter to be the same
############
obs_original = GPObservationSingleOriginal(K=K, D=D, x_grids=x_grids, y_grids=y_grids,
                                           bounds=bounds, mus_init=mus_init, rs=rs, train_rs=train_rs, device=device)
obs_original.us = obs.us
model_2 = HMM(K=K, D=D, M=M, transition=transition, observation=obs_original,
              transition_kwargs=transition_kwargs, device=device)


print(model.log_likelihood(training_data))
print(model_2.log_likelihood(training_data))
#print(obs.log_prob(training_data))
#print(obs_original.log_prob(training_data))

