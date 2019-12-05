import torch
import numpy as np

from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction
from project_ssms.gp_observation import GPObservation, kernel_distsq_doubled
from project_ssms.utils import k_step_prediction_for_gpmodel


import matplotlib.pyplot as plt

def test_model():
    torch.manual_seed(0)
    np.random.seed(0)

    T = 5
    x_grids = np.array([0.0,  10.0])
    y_grids = np.array([0.0,  8.0])

    bounds = np.array([[0.0, 10.0], [0.0,8.0], [0.0, 10.0], [0.0, 8.0]])
    data = np.array([[1.0, 1.0, 1.0, 6.0], [3.0, 6.0, 8.0, 6.0],
                     [4.0, 7.0, 8.0, 5.0], [6.0, 7.0, 5.0, 6.0], [8.0, 2.0, 6.0, 1.0]])
    data = torch.tensor(data, dtype=torch.float64)

    K = 3
    D = 4
    M = 0

    obs = GPObservation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, bounds=bounds, train_rs=True)

    correct_kerneldist_gg = torch.tensor([[  0.,   0.,  64.,  64., 100., 100., 164., 164.],
        [  0.,   0.,  64.,  64., 100., 100., 164., 164.],
        [ 64.,  64.,   0.,   0., 164., 164., 100., 100.],
        [ 64.,  64.,   0.,   0., 164., 164., 100., 100.],
        [100., 100., 164., 164.,   0.,   0.,  64.,  64.],
        [100., 100., 164., 164.,   0.,   0.,  64.,  64.],
        [164., 164., 100., 100.,  64.,  64.,   0.,   0.],
        [164., 164., 100., 100.,  64.,  64.,   0.,   0.]], dtype=torch.float64)
    assert torch.all(torch.eq(correct_kerneldist_gg, obs.kernel_distsq_gg)), obs.kernel_distsq_gg

    log_prob_nocache = obs.log_prob(data)
    print("log_prob_nocache = {}".format(log_prob_nocache))

    kernel_distsq_xg_a = kernel_distsq_doubled(data[:-1, 0:2], obs.inducing_points)
    kernel_distsq_xg_b = kernel_distsq_doubled(data[:-1, 2:4], obs.inducing_points)

    correct_kernel_distsq_xg_a = torch.tensor([[  2.,   2.,  50.,  50.,  82.,  82., 130., 130.],
        [  2.,   2.,  50.,  50.,  82.,  82., 130., 130.],
        [ 45.,  45.,  13.,  13.,  85.,  85.,  53.,  53.],
        [ 45.,  45.,  13.,  13.,  85.,  85.,  53.,  53.],
        [ 65.,  65.,  17.,  17.,  85.,  85.,  37.,  37.],
        [ 65.,  65.,  17.,  17.,  85.,  85.,  37.,  37.],
        [ 85.,  85.,  37.,  37.,  65.,  65.,  17.,  17.],
        [ 85.,  85.,  37.,  37.,  65.,  65.,  17.,  17.]], dtype=torch.float64)
    assert torch.all(torch.eq(correct_kernel_distsq_xg_a, kernel_distsq_xg_a)), kernel_distsq_xg_a

    memory_kwargs = dict(kernel_distsq_xg_a=kernel_distsq_xg_a, kernel_distsq_xg_b=kernel_distsq_xg_b)

    log_prob = obs.log_prob(data, **memory_kwargs)
    print("log_prob = {}".format(log_prob))

    assert torch.all(torch.eq(log_prob_nocache, log_prob))

    Sigma_a, A_a = obs.get_gp_cache(data[:-1, 0:2], 0, **memory_kwargs)
    Sigma_b, A_b = obs.get_gp_cache(data[:-1, 2:4], 1, **memory_kwargs)
    memory_kwargs_2 = dict(Sigma_a=Sigma_a, A_a=A_a, Sigma_b=Sigma_b, A_b=A_b)

    print("calculating log prob 2...")
    log_prob2 = obs.log_prob(data, **memory_kwargs_2)
    assert torch.all(torch.eq(log_prob, log_prob2))

    model = HMM(K=K, D=D, M=M, transition="stationary", observation=obs)
    model.observation.mus_init = data[0] * torch.ones(K, D, dtype=torch.float64)

    # fit
    losses, opt = model.fit(data, optimizer=None, method='adam', num_iters=100, lr=0.01,
                            pbar_update_interval=10,
                            **memory_kwargs)

    plt.figure()
    plt.plot(losses)
    plt.show()

    # most-likely-z
    print("Most likely z...")
    z = model.most_likely_states(data, **memory_kwargs)


    # prediction
    print("0 step prediction")
    if data.shape[0] <= 1000:
        data_to_predict = data
    else:
        data_to_predict = data[-1000:]
    x_predict = k_step_prediction_for_gpmodel(model, z, data_to_predict,
                                              **memory_kwargs)
    x_predict_err = np.mean(np.abs(x_predict - data_to_predict.numpy()), axis=0)

    print("2 step prediction")
    x_predict_2 = k_step_prediction(model, z, data_to_predict, k=2)
    x_predict_2_err = np.mean(np.abs(x_predict_2 - data_to_predict[2:].numpy()), axis=0)

    # samples
    sample_T = 5
    sample_z, sample_x = model.sample(sample_T)

test_model()
