from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.single_transformations.grid_single_transformation import GridSingleTransformation
from project_ssms.feature_funcs import f_corner_vec_func
from project_ssms.momentum_utils import filter_traj_by_speed
from project_ssms.utils import k_step_prediction_for_grid_model, downsample
from project_ssms.plot_utils import plot_1_mice, plot_z
from project_ssms.grid_utils import plot_weights, plot_dynamics, plot_quiver, add_grid
from project_ssms.constants import ARENA_XMIN, ARENA_XMAX, ARENA_YMIN, ARENA_YMAX

from saver.rslts_saving import addDateTime, NumpyEncoder

import torch
import numpy as np
import matplotlib.pyplot as plt

import joblib
import git
import os
import click
import json

################### specifying default arguments ################

@click.command()
@click.option('--job_name', default=None, help='name of the job')
@click.option('--downsample_n', default=1, help='downsample factor. Data size will reduce to 1/downsample_n')
@click.option('--load_model', default=False, help='Whether to load the (trained) model')
@click.option('--load_model_dir', default="", help='Directory of model to load')
@click.option('--train_model', default=True, help='Whether to train the model')
@click.option('--pbar_update_interval', default=500, help='progress bar update interval')
@click.option('--video_clips', default="0,1", help='The starting and ending video clip of the training data')
@click.option('--torch_seed', default=0, help='torch random seed')
@click.option('--np_seed', default=0, help='numpy random seed')
@click.option('--k', default=4, help='number of hidden states')
@click.option('--n_x', default=3, help='number of grids in x axis')
@click.option('--n_y', default=3, help='number of grids in y_axis')
@click.option('--list_of_num_iters', default='5000,5000', help='a list of checkpoint numbers of iterations for training')
@click.option('--list_of_lr', default='0.005,0.005', help='learning rate for training')
@click.option('--sample_t', default=1000, help='length of samples')
def main(job_name, downsample_n, load_model, load_model_dir, train_model, pbar_update_interval,
         video_clips, torch_seed, np_seed, k, n_x, n_y,
         list_of_num_iters, list_of_lr, sample_t):
    if job_name is None:
        raise ValueError("Please provide the job name.")
    K = k
    sample_T = sample_t
    video_clip_start, video_clip_end = [int(x) for x in video_clips.split(",")]
    list_of_num_iters = [int(x) for x in list_of_num_iters.split(",")]
    list_of_lr = [float(x) for x in list_of_lr.split(",")]
    assert len(list_of_num_iters) == len(list_of_lr), "Length of list_of_num_iters must match length of list-of_lr."

    repo = git.Repo('.', search_parent_directories=True)  # SocialBehaviorectories=True)
    repo_dir = repo.working_tree_dir  # SocialBehavior

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)

    ########################## data ########################
    data_dir = repo_dir + '/SocialBehaviorptc/data/trajs_all'
    trajs = joblib.load(data_dir)

    traj = trajs[36000*video_clip_start:36000*video_clip_end]
    traj = downsample(traj, downsample_n)
    f_traj = filter_traj_by_speed(traj, q1=0.99, q2=0.99)

    # only consider 2-dimensional so far
    data = torch.tensor(f_traj[:,:2], dtype=torch.float64)

    ######################### model ####################

    # grids
    x_grid_gap = (ARENA_XMAX - ARENA_XMIN) / n_x
    y_grid_gap = (ARENA_YMAX - ARENA_YMIN) / n_y

    x_grids = [ARENA_XMIN + i * x_grid_gap for i in range(n_x + 1)]
    y_grids = [ARENA_YMIN + i * y_grid_gap for i in range(n_y + 1)]

    bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])

    G = n_x * n_y

    # model
    D = 2
    M = 0
    Df = 4

    if load_model:
        model = joblib.load(load_model_dir)
        tran = model.observation.transformation

        K = model.K
        n_x = len(tran.x_grids) - 1
        n_y = len(tran.y_grids) - 1

    else:
        tran = GridSingleTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, single_transformation="direction",
                              Df=Df, feature_vec_func=f_corner_vec_func, acc_factor=10)
        obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=1, bounds=bounds, transformation=tran)

        model = HMM(K=K, D=D, M=M, observation=obs)
        model.observation.mus_init = data[0] * torch.ones(K, D, dtype=torch.float64)

    # save experiment params
    exp_params = {"job_name":   job_name,
                  'downsample_n': downsample_n,
                  "load_model": load_model,
                  "load_model_dir": load_model_dir,
                  "train_model": train_model,
                  "pbar_update_interval": pbar_update_interval,
                  "K": K,
                  "n_x": n_x,
                  "n_y": n_y,
                  "list_of_num_iters": list_of_num_iters,
                  "list_of_lr": list_of_lr,
                  "video_clip_start": video_clip_start,
                  "video_clip_end": video_clip_end,
                  "sample_T": sample_T}

    print("Experiment params:")
    print(exp_params)

    rslt_dir = addDateTime("rslts/" + job_name)
    rslt_dir = os.path.join(repo_dir, rslt_dir)
    if not os.path.exists(rslt_dir):
        os.makedirs(rslt_dir)
        print("Making result directory...")
    print("Saving to rlst_dir: ", rslt_dir)
    with open(rslt_dir+"/exp_params.json", "w") as f:
        json.dump(exp_params, f, indent=4, cls=NumpyEncoder)

    # compute memories
    masks_a = tran.get_masks(data[:-1])
    feature_vecs_a = f_corner_vec_func(data[:-1])

    m_kwargs_a = dict(feature_vecs=feature_vecs_a)

    ##################### training ############################
    if train_model:
        print("start training")
        list_of_losses = []
        opt = None
        for i, (num_iters, lr) in enumerate(zip(list_of_num_iters, list_of_lr)):
            losses, opt = model.fit(data, optimizer=opt, method='adam', num_iters=num_iters, lr=lr, masks_a=masks_a,
                                    pbar_update_interval=pbar_update_interval, memory_kwargs_a=m_kwargs_a)
            list_of_losses.append(losses)
            # save model
            joblib.dump(model, rslt_dir+"/model_checkpoint{}".format(i))

    #################### inference ###########################

    print("\ninferiring most likely states...")
    z = model.most_likely_states(data, masks_a=masks_a, memory_kwargs_a=m_kwargs_a)

    print("0 step prediction")
    if data.shape[0] <= 1000:
        data_to_predict = data
    else:
        data_to_predict = data[-1000:]
    x_predict = k_step_prediction_for_grid_model(model, z, data_to_predict, memory_kwargs_a=m_kwargs_a)
    x_predict_err = np.mean(np.abs(x_predict - data_to_predict.numpy()), axis=0)

    print("5 step prediction")
    x_predict_5 = k_step_prediction(model, z, data_to_predict, k=5)
    x_predict_5_err = np.mean(np.abs(x_predict_5-data_to_predict[5:].numpy()), axis=0)


    ################### samples #########################

    sample_z, sample_x = model.sample(sample_T)

    center_z = torch.tensor([0], dtype=torch.int)
    center_x = torch.tensor([[150, 190]], dtype=torch.float64)
    sample_z_center, sample_x_center = model.sample(sample_T, prefix=(center_z, center_x))


    ################## dynamics #####################

    # weights
    weights_a = np.array([t.weights.detach().numpy() for t in tran.transformations_a])

    # dynamics
    grid_centers = np.array([[1/2*(x_grids[i] + x_grids[i+1]), 1/2*(y_grids[j] + y_grids[j+1])]
                             for i in range(n_x) for j in range(n_y)])
    unit_corner_vecs = f_corner_vec_func(torch.tensor(grid_centers, dtype=torch.float64))
    unit_corner_vecs = unit_corner_vecs.numpy()
    # (G, 1, Df, d) * (G, K, Df, 1) --> (G, K, Df, d)
    weighted_corner_vecs_a = unit_corner_vecs[:, None] * weights_a[..., None]

    # (G, T-1) For each grid, 0 -- no data in that grid, 1 -- k=0, 2 -- k=1, ... K -- k=K-1
    masks_z_a = np.array([(z[:-1] + 1) * masks_a[g].numpy() for g in range(G)])
    # (G, K) For each grid g, number of data in that grid = k
    grid_z_a = np.array([[sum(masks_z_a[g] == k) for k in range(1,K+1)] for g in range(G)])
    grid_z_a_percentage = grid_z_a / (grid_z_a.sum(axis=1)[:,None] + 1e-6)

    # quiver
    XX, YY = np.meshgrid(np.linspace(20, 310, 30),
                         np.linspace(0, 380, 30))
    XY_grids = np.column_stack((np.ravel(XX), np.ravel(YY))) # shape (900,2) grid values

    XY_next = tran.transform(torch.tensor(XY_grids, dtype=torch.float64)) # (900, 2)
    dXY = XY_next.detach().numpy() - XY_grids[:, None]  # (900, 2)

    #################### saving ##############################

    print("begin saving...")

    if train_model:
        joblib.dump(opt, rslt_dir+"/optimizer")

    # save numbers
    saving_dict = {"z": z, "x_predict": x_predict, "x_predict_5": x_predict_5,
                   "x_predict_err": x_predict_err, "x_predict_5_err": x_predict_5_err,
                   "sample_z": sample_z, "sample_x": sample_x,
                   "sample_z_center": sample_z_center, "sample_x_center": sample_x_center,
                   "grid_z_a_percentage": grid_z_a_percentage}

    if train_model:
        saving_dict['list_of_losses'] = list_of_losses
        for i, losses in enumerate(list_of_losses):
            plt.figure()
            plt.plot(losses)
            plt.savefig(rslt_dir+"/losses_{}.jpg".format(i))
    joblib.dump(saving_dict, rslt_dir+"/numbers")

    # save figures
    plot_z(z)
    plt.savefig(rslt_dir + "/z.jpg")

    plot_z(sample_z)
    plt.savefig(rslt_dir + "/sample_z.jpg")

    plot_z(sample_z_center)
    plt.savefig(rslt_dir + "/sample_z_center.jpg")

    plt.figure(figsize=(4,4))
    plot_1_mice(sample_x)
    plt.legend()
    add_grid(x_grids, y_grids)
    plt.savefig(rslt_dir+"/sample_x.jpg")
    plt.figure(figsize=(4,4))
    plot_1_mice(sample_x_center)
    plt.legend()
    add_grid(x_grids, y_grids)
    plt.savefig(rslt_dir+"/sample_x_center.jpg")

    plot_weights(weights_a, Df, K, x_grids, y_grids, max_weight=tran.transformations_a[0].acc_factor)
    plt.savefig(rslt_dir+"/weights_a.jpg")

    plot_dynamics(weighted_corner_vecs_a, "virgin", x_grids, y_grids, K=K, scale=0.2, percentage=grid_z_a_percentage)
    plt.savefig(rslt_dir+"/dynamics_a.jpg")

    plot_quiver(XY_grids, dXY, 'virgin', K=K, scale=0.2, alpha=0.9)
    plt.savefig(rslt_dir+"/quiver_a.jpg")

    print("Finish running!")


if __name__ == "__main__":
    main()
