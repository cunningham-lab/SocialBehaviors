from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_truncated_normal_observation import ARTruncatedNormalObservation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.utils import find_permutation, random_rotation, k_step_prediction

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

import time

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from hips.plotting.colormaps import gradient_cmap, white_to_color_cmap
color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)


import joblib

datasets_processed = joblib.load('/Users/leah/Columbia/courses/19summer/SocialBehavior/tracedata/all_data_3_1')  # a list of length 30, each is a social_dataset

session_data = datasets_processed[0].render_trajectories([3, 8])
traj0 = np.concatenate((session_data), axis=1)

data = torch.tensor(traj0[:,2:4], dtype=torch.float64)

arena_xmin = 10
arena_xmax = 310

arena_ymin = -5
arena_ymax = 390

bounds = np.array([[arena_xmin, arena_xmax], [arena_ymin, arena_ymax]])

K = 2
D = 1
T = 36000

lags = 1

bounds_1 = bounds[0:1,]

As = np.array([np.column_stack([np.identity(D), np.zeros((D, (lags-1) * D))]) for _ in range(K)])

torch.manual_seed(0)
np.random.seed(0)

tran_1 = LinearTransformation(K=K, D=D, lags=lags, As=As)
observation_1 = ARTruncatedNormalObservation(K=K, D=D, M=0, transformation=tran_1, bounds=bounds_1)

model_1 = HMM(K=K, D=D, M=0, observation=observation_1)

data_1 = data[:,0:1]

#losses_1, optimizer_1 = model_1.fit(data_1, method='adam', num_iters=2000, lr=0.001)

z_1 = model_1.most_likely_states(data_1)

x_predict_arr_lag1 = k_step_prediction(model_1, z_1, data_1)