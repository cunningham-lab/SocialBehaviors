import torch
import numpy as np

from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

from project_ssms.single_transformations.single_momentum_direction_transformation import SingleMomentumDirectionTransformation
from project_ssms.coupled_transformations.grid_transformation import GridTransformation
from project_ssms.momentum_utils import filter_traj_by_speed, get_momentum_in_batch, get_momentum
from project_ssms.feature_funcs import unit_vector_to_fixed_loc, unit_vector_to_other, \
    feature_vec_func, feature_direction_vec
from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.utils import k_step_prediction_for_grid_model

import joblib
import git


repo = git.Repo('.', search_parent_directories=True) # SocialBehaviorectories=True)
repo_dir = repo.working_tree_dir  # SocialBehavior

torch.manual_seed(0)
np.random.seed(0)

"""
# fake data
T= 2
x_grids = np.array([0.0, 5.0, 10.0])
y_grids = np.array([0.0, 4.0, 8.0])

data = np.array([[1.0, 1.0, 1.0, 6.0], [3.0, 6.0, 8.0, 6.0],
                 [4.0, 7.0, 8.0, 5.0], [6.0, 7.0, 5.0, 6.0], [8.0, 2.0, 6.0, 1.0]])
data = torch.tensor(data, dtype=torch.float64)


def toy_feature_vec_func(s, o):
 
    feature_funcs = [unit_vector_to_other,
                     lambda s, o: unit_vector_to_fixed_loc(s, o, torch.tensor([0, 0], dtype=torch.float64)),
                     lambda s, o: unit_vector_to_fixed_loc(s, o, torch.tensor([0, 8], dtype=torch.float64)),
                     lambda s, o: unit_vector_to_fixed_loc(s, o, torch.tensor([10, 0], dtype=torch.float64)),
                     lambda s, o: unit_vector_to_fixed_loc(s, o, torch.tensor([10, 8], dtype=torch.float64)),
                     ]

    features = [f(s, o) for f in feature_funcs]  # each is a tensor of shape (T,2), and there are Df items of them
    features = torch.stack(features, dim=1)
    return features

K = 3
D = 4
M = 0

Df = 5
momentum_lags = 30
momentum_weights = np.arange(0.55, 2.05, 0.05)
momentum_weights = torch.tensor(momentum_weights, dtype=torch.float64)

bounds = np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 10.0], [0.0, 8.0]])


tran = GridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids,
                          Df=Df, feature_vec_func=toy_feature_vec_func,
                          lags=momentum_lags, momentum_weights=momentum_weights)

# compute memories
masks_a, masks_b = tran.get_masks(data[:-1])

momentum_vecs_a = get_momentum_in_batch(data[:-1, 0:2], lags=momentum_lags, weights=momentum_weights)
momentum_vecs_b = get_momentum_in_batch(data[:-1, 2:4], lags=momentum_lags, weights=momentum_weights)

feature_vecs_a = toy_feature_vec_func(data[:-1, 0:2], data[:-1, 2:4])
feature_vecs_b = toy_feature_vec_func(data[:-1, 2:4], data[:-1, 0:2])

m_kwargs_a = dict(momentum_vecs=momentum_vecs_a, feature_vecs=feature_vecs_a)
m_kwargs_b = dict(momentum_vecs=momentum_vecs_b, feature_vecs=feature_vecs_b)

# observation
obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=momentum_lags, bounds=bounds, transformation=tran)

# model
model = HMM(K=K, D=D, M=M, observation=obs)


################## test params #######################
tran2 = tran = GridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, 
                            single_transformation="momentum_direction",
                          Df=Df, feature_vec_func=toy_feature_vec_func,
                          lags=momentum_lags, momentum_weights=momentum_weights)

obs2 = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=momentum_lags, bounds=bounds, transformation=tran2)

model2 = HMM(K=K, D=D, M=M, observation=obs2)

model2.params = model.params

for p1, p2 in zip(model.params_unpack, model2.params_unpack):
    assert torch.all(torch.eq(p1, p2))


################# check memory ###############
log_prob = model.log_likelihood(data)
log_prob_2 = model.log_likelihood(data, masks=(masks_a, masks_b),
                                  memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
assert torch.eq(log_prob, log_prob_2)


# training
print("start training...")
num_iters = 10
losses, opt = model.fit(data, num_iters=num_iters, lr=0.001, masks=(masks_a, masks_b),
                                  memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
# sampling
print("start sampling")
sample_z, sample_x = model.sample(30)


# inference
print("inferiring most likely states...")
z = model.most_likely_states(data, masks=(masks_a, masks_b),
                                  memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)


print("0 step prediction")
x_predict = k_step_prediction_for_grid_model(model, z, data, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)

print("k step prediction")
x_predict_2 = k_step_prediction(model, z, data, 2)

"""


############# Real data ##############

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

bounds = np.array([[arena_xmin - 5, arena_xmax + 5], [arena_ymin - 5, arena_ymax + 5],
                   [arena_xmin - 5, arena_xmax + 5], [arena_ymin - 5, arena_ymax + 5]])

momentum_lags = 30
momentum_weights = np.arange(0.55, 2.05, 0.05)
momentum_weights = torch.tensor(momentum_weights, dtype=torch.float64)

K = 4
D = 4
M = 0

T = data.shape[0]


# make 3 by 3 grid world
x_grid_gap = (bounds[0][1] - bounds[0][0]) / 3
y_grid_gap = (bounds[1][1] - bounds[1][0]) / 3

x_grids = np.array([bounds[0][0] + i * x_grid_gap for i in range(4)])
y_grids = np.array([bounds[1][0] + i * y_grid_gap for i in range(4)])

"""
Df = 5

tran = GridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, single_transformation="momentum_direction",
                          Df=Df, feature_vec_func=feature_vec_func,
                          lags=momentum_lags, momentum_weights=momentum_weights)



# compute memories
masks_a, masks_b = tran.get_masks(data[:-1])


momentum_vecs_a = get_momentum_in_batch(data[:-1, 0:2], lags=momentum_lags, weights=momentum_weights)
momentum_vecs_b = get_momentum_in_batch(data[:-1, 2:4], lags=momentum_lags, weights=momentum_weights)

feature_vecs_a = feature_vec_func(data[:-1, 0:2], data[:-1, 2:4])
feature_vecs_b = feature_vec_func(data[:-1, 2:4], data[:-1, 0:2])


m_kwargs_a = dict(momentum_vecs=momentum_vecs_a, feature_vecs=feature_vecs_a)
m_kwargs_b = dict(momentum_vecs=momentum_vecs_b, feature_vecs=feature_vecs_b)


# observation
obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=momentum_lags, bounds=bounds, transformation=tran)


# model
model = HMM(K=K, D=D, M=M, observation=obs)

log_prob = model.log_likelihood(data, masks=(masks_a, masks_b),
                                  memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
log_prob_2 = model.log_likelihood(data)
assert torch.eq(log_prob, log_prob_2)
print(log_prob)

# training
print("start training...")
num_iters = 10
losses, opt = model.fit(data, num_iters=num_iters, lr=0.001, masks=(masks_a, masks_b),
                                  memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)

# sampling
print("start sampling")
sample_z, sample_x = model.sample(T)


# inference
print("inferiring most likely states...")
z = model.most_likely_states(data, masks=(masks_a, masks_b),
                                  memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)


print("0 step prediction")
x_predict = k_step_prediction_for_grid_model(model, z, data, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)

print("k step prediction")
x_predict_10 = k_step_prediction(model, z, data, 10)

"""

Df = 4

CORNERS = torch.tensor([[0,0], [0, 330], [300, 0], [300, 330]], dtype=torch.float64)

f_corner_vec_func = lambda s: feature_direction_vec(s, CORNERS)

tran = GridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, single_transformation="direction",
                          Df=Df, feature_vec_func=f_corner_vec_func)

# compute memories
masks_a, masks_b = tran.get_masks(data[:-1])
feature_vecs_a = f_corner_vec_func(data[:-1, 0:2])
feature_vecs_b = f_corner_vec_func(data[:-1, 2:4])

m_kwargs_a = dict(feature_vecs=feature_vecs_a)
m_kwargs_b = dict(feature_vecs=feature_vecs_b)

obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=1, bounds=bounds, transformation=tran)

# model
model = HMM(K=K, D=D, M=M, observation=obs)

log_prob = model.log_likelihood(data, masks=(masks_a, masks_b),
                                  memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)

log_prob_2 = model.log_likelihood(data)
assert torch.eq(log_prob, log_prob_2)
print(log_prob)

# training
print("start training...")
num_iters = 10
losses, opt = model.fit(data, num_iters=num_iters, lr=0.001, masks=(masks_a, masks_b),
                                  memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)

# sampling
print("start sampling")
sample_z, sample_x = model.sample(T)

print("start sampling based on transformation")
sample_z2, sample_x2 = model.sample(T, transformation=True)


# inference
print("inferiring most likely states...")
z = model.most_likely_states(data, masks=(masks_a, masks_b),
                                  memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)


print("0 step prediction")
x_predict = k_step_prediction_for_grid_model(model, z, data, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)

print("k step prediction")
x_predict_10 = k_step_prediction(model, z, data, 10)