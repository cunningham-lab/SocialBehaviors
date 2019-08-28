import torch
import numpy as np
import matplotlib.pyplot as plt

from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

from project_ssms.coupled_transformations.space_dependent_transformation import SpaceDependentTransformation
from project_ssms.feature_funcs import feature_direction_vec, f_corner_vec_func
from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.utils import k_step_prediction_for_artn_model, downsample
from project_ssms.constants import ARENA_XMIN, ARENA_XMAX, ARENA_YMIN, ARENA_YMAX
from project_ssms.grid_utils import plot_realdata_quiver

import joblib
import git

repo = git.Repo('.', search_parent_directories=True)  # SocialBehaviorectories=True)
repo_dir = repo.working_tree_dir  # SocialBehavior

torch.manual_seed(0)
np.random.seed(0)

"""
# fake data
T = 6

data = np.array([[1.0, 1.0, 1.0, 6.0], [3.0, 6.0, 8.0, 6.0],
                 [4.0, 7.0, 8.0, 5.0], [6.0, 7.0, 5.0, 6.0],
                 [8.0, 2.0, 6.0, 1.0], [4.0, 8.0, 6.0, 2.0]])
data = torch.tensor(data, dtype=torch.float64)


corners = [torch.tensor([0,0], dtype=torch.float64), torch.tensor([0, 8], dtype=torch.float64),
           torch.tensor([10,0], dtype=torch.float64), torch.tensor([10,8], dtype=torch.float64)]


def toy_feature_vec_func(s):
    return feature_direction_vec(s, corners)


K = 3
D = 4
M = 0

Df = 4

bounds = np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 10.0], [0.0, 8.0]])


tran = SpaceDependentTransformation(K=K, D=D, Df=Df, feature_vec_func=toy_feature_vec_func, dhs=[5], acc_factor=10)

# compute memories

feature_vecs_a = toy_feature_vec_func(data[:-1, 0:2])
feature_vecs_b = toy_feature_vec_func(data[:-1, 2:4])

m_kwargs_a = dict(feature_vecs=feature_vecs_a)
m_kwargs_b = dict(feature_vecs=feature_vecs_b)

# observation
obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=1, bounds=bounds, transformation=tran)

# model
model = HMM(K=K, D=D, M=M, observation=obs)


################## test params #######################
tran2 = SpaceDependentTransformation(K=K, D=D, Df=Df, feature_vec_func=toy_feature_vec_func, dhs=[5], acc_factor=10)

obs2 = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=1, bounds=bounds, transformation=tran2)
model2 = HMM(K=K, D=D, M=M, observation=obs2)

model2.params = model.params

for p1, p2 in zip(model.params_unpack, model2.params_unpack):
    assert torch.all(torch.eq(p1, p2))


################# check memory ###############
log_prob = model.log_likelihood(data)
log_prob_2 = model.log_likelihood(data, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
assert torch.eq(log_prob, log_prob_2)


# training
print("start training...")
num_iters = 10
losses, opt = model.fit(data, num_iters=num_iters, lr=0.001, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
# sampling
print("start sampling")
sample_z, sample_x = model.sample(30)


# inference
print("inferiring most likely states...")
z = model.most_likely_states(data, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)


print("0 step prediction")
x_predict = k_step_prediction_for_artn_model(model, z, data, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)

print("k step prediction")
x_predict_2 = k_step_prediction(model, z, data, 2)

"""

############# Real data ##############

data_dir = repo_dir + '/SocialBehaviorptc/data/trajs_all'
trajs = joblib.load(data_dir)

traj0 = trajs[36000 * 0:36000 * 1]
traj0 = downsample(traj0, 2)

data = torch.tensor(traj0, dtype=torch.float64)

bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX],
                   [ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])

K = 4
D = 4
M = 0

T = data.shape[0]

Df = 4

# compute memories

feature_vecs_a = f_corner_vec_func(data[:-1, 0:2])
feature_vecs_b = f_corner_vec_func(data[:-1, 2:4])

m_kwargs_a = dict(feature_vecs=feature_vecs_a)
m_kwargs_b = dict(feature_vecs=feature_vecs_b)


model = joblib.load("model")
"""
tran = SpaceDependentTransformation(K=K, D=D, Df=Df, feature_vec_func=f_corner_vec_func, dhs=[8,8], acc_factor=20)
obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=1, bounds=bounds, transformation=tran)
model = HMM(K=K, D=D, M=M, observation=obs)
"""

log_prob = model.log_likelihood(data, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
#log_prob_2 = model.log_likelihood(data)
#assert torch.eq(log_prob, log_prob_2)
print(log_prob)

# training
print("start training...")
num_iters = 3000
losses, opt = model.fit(data, num_iters=num_iters, lr=0.01, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
joblib.dump(model, "model")
joblib.dump(losses, "losses")
plt.plot(losses)
# sampling
print("start sampling")
sample_z, sample_x = model.sample(T)
plot_realdata_quiver(sample_x, scale=1, title="sample")
plt.show()
joblib.dump((sample_z, sample_x), "samples")

# inference
print("inferiring most likely states...")
z = model.most_likely_states(data, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
joblib.dump(z, "z")

data_to_predict = data[-1000:]
print("0 step prediction")
x_predict = k_step_prediction_for_artn_model(model, z, data_to_predict, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
err = np.mean(np.abs(x_predict - data_to_predict.numpy()), axis=0)
print(err)

print("k step prediction")
x_predict_5 = k_step_prediction(model, z, data_to_predict, 5)
err = np.mean(np.abs(x_predict_5 - data_to_predict[5:].numpy()), axis=0)
print(err)

