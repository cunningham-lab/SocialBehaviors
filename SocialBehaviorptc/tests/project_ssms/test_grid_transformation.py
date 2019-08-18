import torch
import numpy as np

from project_ssms.unit_transformations.unit_momentum_direction_transformation import SingleMomentumDirectionTransformation
from project_ssms.coupled_transformations.grid_transformation import GridTransformation
from project_ssms.momentum_utils import filter_traj_by_speed, get_momentum_in_batch, get_momentum
from project_ssms.feature_funcs import feature_vec_func, unit_vector_to_fixed_loc, unit_vector_to_other

import joblib


torch.manual_seed(0)
np.random.seed(0)

"""
# real data
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
data = data[:2]

ARENA_XMIN = 0
ARENA_XMAX = 330

ARENA_YMIN = -10
ARENA_YMAX = 390

# make 3 by 3 grid world
x_grid_gap = (ARENA_XMAX - ARENA_XMIN) / 2
y_grid_gap = (ARENA_YMAX - ARENA_YMIN) / 2

x_grids = np.array([ARENA_XMIN + i * x_grid_gap for i in range(3)])
y_grids = np.array([ARENA_YMIN + j * y_grid_gap for j in range(3)])

"""
# fake data
T= 2
x_grids = np.array([0.0, 5.0, 10.0])
y_grids = np.array([0.0, 4.0, 8.0])

data = np.array([[1.0, 1.0, 1.0, 6.0], [3.0, 6.0, 8.0, 6.0],
                 [4.0, 7.0, 8.0, 5.0], [6.0, 7.0, 5.0, 6.0], [8.0, 2.0, 6.0, 1.0]])
data = torch.tensor(data, dtype=torch.float64)


def toy_feature_vec_func(s, o):
    """
    :param s: self, (T, 2)
    :param o: other, (T, 2)
    :return: features, (T, Df, 2)
    """
    feature_funcs = [unit_vector_to_other,
                     lambda s, o: unit_vector_to_fixed_loc(s, torch.tensor([0, 0], dtype=torch.float64)),
                     lambda s, o: unit_vector_to_fixed_loc(s, torch.tensor([0, 8], dtype=torch.float64)),
                     lambda s, o: unit_vector_to_fixed_loc(s, torch.tensor([10, 0], dtype=torch.float64)),
                     lambda s, o: unit_vector_to_fixed_loc(s, torch.tensor([10, 8], dtype=torch.float64)),
                     ]

    features = [f(s, o) for f in feature_funcs]  # each is a tensor of shape (T,2), and there are Df items of them
    features = torch.stack(features, dim=1)
    return features

K = 3
D = 4

Df = 5
momentum_lags = 30
momentum_weights = np.arange(0.55, 2.05, 0.05)
momentum_weights = torch.tensor(momentum_weights, dtype=torch.float64)


tran = GridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, single_transformation="momentum_direction",
                          Df=Df, feature_vec_func=toy_feature_vec_func,
                          lags=momentum_lags, momentum_weights=momentum_weights)

assert tran.G == 4
assert len(tran.params) == 8


################# check parameters ######################
tran_2 = GridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, single_transformation="momentum_direction",
                          Df=Df, feature_vec_func=toy_feature_vec_func,
                          lags=momentum_lags, momentum_weights=momentum_weights)

tran_2.params = tran.params

for t1, t2 in zip(tran.params, tran_2.params):
    assert torch.all(torch.eq(t1, t2))

################## check mask ##########################
masks_a, masks_b = tran.get_masks(data)

true_masks_a = [torch.tensor([1, 0, 0, 0, 0], dtype=torch.float64),
                   torch.tensor([0, 1, 1, 0, 0], dtype=torch.float64),
                   torch.tensor([0, 0, 0, 0, 1], dtype=torch.float64),
                   torch.tensor([0, 0, 0, 1, 0], dtype=torch.float64)]
true_masks_b = [torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64),
                   torch.tensor([1, 0, 0, 1, 0], dtype=torch.float64),
                   torch.tensor([0, 0, 0, 0, 1], dtype=torch.float64),
                   torch.tensor([0, 1, 1, 0, 0], dtype=torch.float64)]

for m, p in zip(masks_a, true_masks_a):
    assert torch.all(torch.eq(m, p))
for m, p in zip(masks_b, true_masks_b):
    assert torch.all(torch.eq(m, p))


################### check transformation ##################
data_transformed = tran.transform(data)


################## check memory #####################

momentum_vecs_a = get_momentum_in_batch(data[:, :2], lags=momentum_lags, weights=momentum_weights)
momentum_vecs_b = get_momentum_in_batch(data[:, 2:], lags=momentum_lags, weights=momentum_weights)

feature_vecs_a = toy_feature_vec_func(data[:, :2], data[:, 2:])
feature_vecs_b = toy_feature_vec_func(data[:, 2:], data[:, :2])

m_kwargs_a = dict(momentum_vecs=momentum_vecs_a, feature_vecs=feature_vecs_a)
m_kwargs_b = dict(momentum_vecs=momentum_vecs_b, feature_vecs=feature_vecs_b)

data_transformed_2 = tran.transform(data, masks=(masks_a, masks_b),
                                    memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)

assert torch.all(torch.eq(data_transformed, data_transformed_2))


############# check transform_condition_on_z #############




