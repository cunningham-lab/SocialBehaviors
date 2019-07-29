from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_gaussian_observation import ARGaussianObservation
from ssm_ptc.transformations.linear import LinearTransformation
from ssm_ptc.utils import find_permutation, random_rotation, k_step_prediction

import torch
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from tqdm import tqdm_notebook as tqdm
import sys
import time

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


npr.seed(0)
torch.manual_seed(0)

K = 3
D = 2
M = 1
T = 100

# Create an exogenous input
inpt = np.sin(2 * np.pi * np.arange(T) / 50)[:, None] + 1e-1 * npr.randn(T, M)
inpt = torch.tensor(inpt, dtype=torch.float64)

true_model = HMM(K=K, D=D, M=M, transition='inputdriven', observation='gaussian')

z, data = true_model.sample(T, input=inpt, return_np=True)

lls = true_model.log_likelihood(data, inpt)