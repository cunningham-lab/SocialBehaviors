from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.coupled_transformations.grid_transformation import GridTransformation
from project_ssms.unit_transformations import unit_direction_transformation
from project_ssms.feature_funcs import f_corner_vec_func
from project_ssms.momentum_utils import filter_traj_by_speed
from project_ssms.utils import k_step_prediction_for_grid_model, downsample
from project_ssms.plot_utils import plot_z, plot_2_mice, plot_4_traces
from project_ssms.grid_utils import plot_weights, plot_dynamics, \
    plot_quiver, add_grid, plot_realdata_quiver, get_z_percentage_by_grid
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
@click.option('--filter_traj', default=False, help='whether or not to filter the trajectory by SPEED')
@click.option('--load_model', default=False, help='Whether to load the (trained) model')
@click.option('--train_model', default=True, help='Whether to train the model')
@click.option('--pbar_update_interval', default=500, help='progress bar update interval')
@click.option('--load_model_dir', default="", help='Directory of model to load')
@click.option('--video_clips', default="0,1", help='The starting video clip of the training data')
@click.option('--torch_seed', default=0, help='torch random seed')
@click.option('--np_seed', default=0, help='numpy random seed')
@click.option('--k', default=4, help='number of hidden states')
@click.option('--n_x', default=3, help='number of grids in x axis')
@click.option('--n_y', default=3, help='number of grids in y_axis')
@click.option('--list_of_num_iters', default='5000,5000', help='a list of checkpoint numbers of iterations for training')
@click.option('--list_of_lr', default='0.005, 0.005', help='learning rate for training')
@click.option('--sample_t', default=100, help='length of samples')
@click.option('--quiver_scale', default=0.3, help='scale for the quiver plots')
def main(job_name, downsample_n, filter_traj, load_model, load_model_dir, train_model, pbar_update_interval, video_clips,
         torch_seed, np_seed, k, n_x, n_y, list_of_num_iters, list_of_lr, sample_t, quiver_scale):
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
    if filter_traj:
        traj = filter_traj_by_speed(traj, q1=0.99, q2=0.99)

    data = torch.tensor(traj, dtype=torch.float64)

    ######################### model ####################

    # grids
    x_grid_gap = (ARENA_XMAX - ARENA_XMIN) / n_x
    y_grid_gap = (ARENA_YMAX - ARENA_YMIN) / n_y

    x_grids = [ARENA_XMIN + i * x_grid_gap for i in range(n_x + 1)]
    y_grids = [ARENA_YMIN + i * y_grid_gap for i in range(n_y + 1)]

    bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX],
                       [ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])

    G = n_x * n_y

    # model
    D = 4
    M = 0
    Df = 4

    if load_model:
        model = joblib.load(load_model_dir)
        tran = model.observation.transformation

        K = model.K
        n_x = len(tran.x_grids) - 1
        n_y = len(tran.y_grids) - 1

    else:
        tran = GridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, unit_transformation="direction",
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
    masks_a, masks_b = tran.get_masks(data[:-1])
    feature_vecs_a = f_corner_vec_func(data[:-1, 0:2])
    feature_vecs_b = f_corner_vec_func(data[:-1, 2:4])

    m_kwargs_a = dict(feature_vecs=feature_vecs_a)
    m_kwargs_b = dict(feature_vecs=feature_vecs_b)


    ##################### training ############################
    if train_model:
        print("start training")
        list_of_losses = []
        opt = None
        for i, (num_iters, lr) in enumerate(zip(list_of_num_iters, list_of_lr)):
            losses, opt = model.fit(data, optimizer=opt, method='adam', num_iters=num_iters, lr=lr,
                                    masks=(masks_a, masks_b), pbar_update_interval=pbar_update_interval,
                                    memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
            list_of_losses.append(losses)
            # save model
            joblib.dump(model, rslt_dir+"/model_checkpoint{}".format(i))

    #################### inference ###########################

    print("\ninferiring most likely states...")
    z = model.most_likely_states(data, masks=(masks_a, masks_b),
                                      memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)

    print("0 step prediction")
    if data.shape[0] <= 1000:
        data_to_predict = data
    else:
        data_to_predict = data[-1000:]
    x_predict = k_step_prediction_for_grid_model(model, z, data_to_predict, memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
    x_predict_err = np.mean(np.abs(x_predict - data_to_predict.numpy()), axis=0)

    print("5 step prediction")
    x_predict_5 = k_step_prediction(model, z, data_to_predict, k=5)
    x_predict_5_err = np.mean(np.abs(x_predict_5-data_to_predict[5:].numpy()), axis=0)


    ################### samples #########################

    sample_z, sample_x = model.sample(sample_T)

    center_z = torch.tensor([0], dtype=torch.int)
    center_x = torch.tensor([[150, 190, 200, 200]], dtype=torch.float64)
    sample_z_center, sample_x_center = model.sample(sample_T, prefix=(center_z, center_x))


    ################## dynamics #####################

    # weights
    weights_a = np.array([t.weights.detach().numpy() for t in tran.transformations_a])
    weights_b = np.array([t.weights.detach().numpy() for t in tran.transformations_b])

    # dynamics
    grid_centers = np.array([[1/2*(x_grids[i] + x_grids[i+1]), 1/2*(y_grids[j] + y_grids[j+1])]
                             for i in range(n_x) for j in range(n_y)])
    unit_corner_vecs = f_corner_vec_func(torch.tensor(grid_centers, dtype=torch.float64))
    unit_corner_vecs = unit_corner_vecs.numpy()
    # (G, 1, Df, d) * (G, K, Df, 1) --> (G, K, Df, d)
    weighted_corner_vecs_a = unit_corner_vecs[:, None] * weights_a[..., None]
    weighted_corner_vecs_b = unit_corner_vecs[:, None] * weights_b[..., None]

    grid_z_a_percentage = get_z_percentage_by_grid(masks_a, z, K, G)
    grid_z_b_percentage = get_z_percentage_by_grid(masks_b, z, K, G)

    # quiver
    XX, YY = np.meshgrid(np.linspace(20, 310, 30),
                         np.linspace(0, 380, 30))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY))) # shape (900,2) grid values
    XY_grids = np.concatenate((XY, XY), axis=1)

    XY_next = tran.transform(torch.tensor(XY_grids, dtype=torch.float64))
    dXY = XY_next.detach().numpy() - XY_grids[:, None]

    #################### saving ##############################

    print("begin saving...")

    if train_model:
        joblib.dump(opt, rslt_dir+"/optimizer")

    # save summary
    avg_transform_speed = np.average(np.abs(dXY), axis=0)
    avg_sample_speed = np.average(np.abs(np.diff(sample_x, axis=0)), axis=0)
    avg_sample_center_speed = np.average(np.abs(np.diff(sample_x_center, axis=0)), axis=0)
    avg_data_speed = np.average(np.abs(np.diff(data.numpy(), axis=0)), axis=0)

    summary_dict = {"x_predict_err": x_predict_err, "x_predict_5_err": x_predict_5_err,
                    "variance": torch.exp(model.observation.log_sigmas).detach().numpy(),
                    "log_likes": model.log_likelihood(data).detach().numpy(),
                    "grid_z_a_percentage": grid_z_a_percentage, "grid_z_b_percentage": grid_z_b_percentage,
                    "avg_transform_speed": avg_transform_speed, "avg_data_speed": avg_data_speed,
                    "avg_sample_speed": avg_sample_speed, "avg_sample_center_speed": avg_sample_center_speed}
    with open(rslt_dir+"/summary.json", "w") as f:
        json.dump(summary_dict, f, indent=4, cls=NumpyEncoder)

    # save numbers
    saving_dict = {"z": z, "x_predict": x_predict, "x_predict_5": x_predict_5,
                   "sample_z": sample_z, "sample_x": sample_x,
                   "sample_z_center": sample_z_center, "sample_x_center": sample_x_center}

    if train_model:
        saving_dict['list_of_losses'] = list_of_losses
        for i, losses in enumerate(list_of_losses):
            plt.figure()
            plt.plot(losses)
            plt.savefig(rslt_dir+"/losses_{}.jpg".format(i))
    joblib.dump(saving_dict, rslt_dir+"/numbers")

    # save figures
    plot_z(z, K)
    plt.savefig(rslt_dir+"/z.jpg")

    if not os.path.exists(rslt_dir+"/samples"):
        os.makedirs(rslt_dir+"/samples")
        print("Making samples directory...")

    plot_z(sample_z, K)
    plt.savefig(rslt_dir+"/samples/sample_z_{}.jpg".format(sample_T))

    plot_z(sample_z_center, K)
    plt.savefig(rslt_dir+"/samples/sample_z_center_{}.jpg".format(sample_T))

    plt.figure(figsize=(4,4))
    plot_2_mice(sample_x)
    plt.legend()
    add_grid(x_grids, y_grids)
    plt.savefig(rslt_dir+"/samples/sample_x_{}.jpg".format(sample_T))
    plt.figure(figsize=(4,4))
    plot_2_mice(sample_x_center)
    plt.legend()
    add_grid(x_grids, y_grids)
    plt.savefig(rslt_dir+"/samples/sample_x_center_{}.jpg".format(sample_T))

    plot_realdata_quiver(sample_x, x_grids, y_grids)
    plt.savefig(rslt_dir+"/samples/sample_x_quiver_{}.jpg".format(sample_T))

    plot_realdata_quiver(sample_x_center, x_grids, y_grids)
    plt.savefig(rslt_dir+"/samples/sample_x_center_quiver_{}.jpg".format(sample_T))

    if not os.path.exists(rslt_dir+"/dynamics"):
        os.makedirs(rslt_dir+"/dynamics")
        print("Making dynamics directory...")

    plot_weights(weights_a, Df, K, x_grids, y_grids, max_weight=tran.transformations_a[0].acc_factor)
    plt.savefig(rslt_dir+"/dynamics/weights_a.jpg")

    plot_weights(weights_b, Df, K, x_grids, y_grids, max_weight=tran.transformations_b[0].acc_factor)
    plt.savefig(rslt_dir+"/dynamics/weights_b.jpg")

    plot_dynamics(weighted_corner_vecs_a, "virgin", x_grids, y_grids, K=K, scale=quiver_scale, percentage=grid_z_a_percentage)
    plt.savefig(rslt_dir+"/dynamics/dynamics_a.jpg")

    plot_dynamics(weighted_corner_vecs_b, "mother", x_grids, y_grids, K=K, scale=quiver_scale, percentage=grid_z_b_percentage)
    plt.savefig(rslt_dir+"/dynamics/dynamics_b.jpg")

    plot_quiver(XY_grids[:, 0:2], dXY[..., 0:2], 'virgin', K=K, scale=quiver_scale, alpha=0.9)
    plt.savefig(rslt_dir+"/dynamics/quiver_a.jpg")

    plot_quiver(XY_grids[:, 2:4], dXY[..., 2:4], 'mother', K=K, scale=quiver_scale, alpha=0.9)
    plt.savefig(rslt_dir+"/dynamics/quiver_b.jpg")

    print("Finish running!")


if __name__ == "__main__":
    main()
