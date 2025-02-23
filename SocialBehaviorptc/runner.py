from ssm_ptc.models.hmm import HMM

from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.coupled_transformations.grid_transformation import GridTransformation
from project_ssms.feature_funcs import f_corner_vec_func
from project_ssms.momentum_utils import filter_traj_by_speed
from project_ssms.utils import downsample
from project_ssms.constants import ARENA_XMIN, ARENA_XMAX, ARENA_YMIN, ARENA_YMAX

from saver.rslts_saving import addDateTime, NumpyEncoder
from saver.runner_grid_rslt_saving import rslt_saving

import torch
import numpy as np

import joblib
import git
import os
import click
import json


################### specifying default arguments ################


@click.command()
@click.option('--job_name', default=None, help='name of the job')
@click.option('--downsample_n', default=1, help='downsample factor. Data size will reduce to 1/downsample_n')
@click.option('--filter_traj', is_flag=True, help='whether or not to filter the trajectory by SPEED')
@click.option('--load_model', is_flag=True, help='Whether to load the (trained) model')
@click.option('--load_model_dir', default="", help='Directory of model to load')
@click.option('--transition', default="stationary", help='type of transition (str)')
@click.option('--sticky_alpha', default=1, help='value of alpha in sticky transition')
@click.option('--sticky_kappa', default=100, help='value of kappa in sticky transition')
@click.option('--acc_factor', default=None, help="acc factor in direction model")
@click.option('--train_model', is_flag=True, help='Whether to train the model')
@click.option('--pbar_update_interval', default=500, help='progress bar update interval')
@click.option('--load_opt_dir', default="", help='Directory of optimizer to load.')
@click.option('--video_clips', default="0,1", help='The starting video clip of the training data')
@click.option('--torch_seed', default=0, help='torch random seed')
@click.option('--np_seed', default=0, help='numpy random seed')
@click.option('--k', default=4, help='number of hidden states. Would be overwritten if load model.')
@click.option('--x_grids', default=None, help='x coordinates to specify the grids')
@click.option('--y_grids', default=None, help='y coordinates to specify the grids')
@click.option('--n_x', default=3, help='number of grids in x axis.'
                                       ' Would be overwritten if x_grids is provided, or load model.')
@click.option('--n_y', default=3, help='number of grids in y_axis.'
                                       ' Would be overwritten if y_grids is provided, or load model.')
@click.option('--list_of_num_iters', default='5000,5000',
              help='a list of checkpoint numbers of iterations for training')
@click.option('--list_of_lr', default='0.005, 0.005', help='learning rate for training')
@click.option('--sample_t', default=100, help='length of samples')
@click.option('--quiver_scale', default=0.5, help='scale for the quiver plots')
def main(job_name, downsample_n, filter_traj, load_model, load_model_dir, load_opt_dir,
         transition, sticky_alpha, sticky_kappa, acc_factor, k, x_grids, y_grids, n_x, n_y,
         train_model,  pbar_update_interval, video_clips, torch_seed, np_seed,
         list_of_num_iters, list_of_lr, sample_t, quiver_scale):
    if job_name is None:
        raise ValueError("Please provide the job name.")
    K = k
    sample_T = sample_t
    video_clip_start, video_clip_end = [int(x) for x in video_clips.split(",")]
    list_of_num_iters = [int(x) for x in list_of_num_iters.split(",")]
    list_of_lr = [float(x) for x in list_of_lr.split(",")]
    assert len(list_of_num_iters) == len(list_of_lr), "Length of list_of_num_iters must match length of list-of_lr."
    for lr in list_of_lr:
        if lr > 1:
            raise ValueError("Learning rate should not be larger than 1!")

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

    # model
    D = 4
    M = 0
    Df = 4

    if load_model:
        print("Loading the model from ", load_model_dir)
        model = joblib.load(load_model_dir)
        tran = model.observation.transformation

        K = model.K

        n_x = len(tran.x_grids) - 1
        n_y = len(tran.y_grids) - 1

        acc_factor = tran.transformations_a[0].acc_factor

    else:
        print("Creating the model...")
        bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX],
                           [ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])

        # grids
        if x_grids is None:
            x_grid_gap = (ARENA_XMAX - ARENA_XMIN) / n_x
            x_grids = [ARENA_XMIN + i * x_grid_gap for i in range(n_x + 1)]
        else:
            x_grids = [float(x) for x in x_grids.split(",")]
            n_x = len(x_grids) - 1

        if y_grids is None:
            y_grid_gap = (ARENA_YMAX - ARENA_YMIN) / n_y
            y_grids = [ARENA_YMIN + i * y_grid_gap for i in range(n_y + 1)]
        else:
            y_grids = [float(x) for x in y_grids.split(",")]
            n_y = len(y_grids) - 1

        if acc_factor is None:
            acc_factor = downsample_n * 10

        tran = GridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, unit_transformation="direction",
                                  Df=Df, feature_vec_func=f_corner_vec_func, acc_factor=acc_factor)
        obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=1, bounds=bounds, transformation=tran)

        if transition == 'sticky':
            transition_kwargs = dict(alpha=sticky_alpha, kappa=sticky_kappa)
        else:
            transition_kwargs = None
        model = HMM(K=K, D=D, M=M, transition=transition, observation=obs, transition_kwargs=transition_kwargs)
        model.observation.mus_init = data[0] * torch.ones(K, D, dtype=torch.float64)

    # save experiment params
    exp_params = {"job_name":   job_name,
                  'downsample_n': downsample_n,
                  "load_model": load_model,
                  "load_model_dir": load_model_dir,
                  "load_opt_dir": load_opt_dir,
                  "transition": transition,
                  "sticky_alpha": sticky_alpha,
                  "sticky_kappa": sticky_kappa,
                  "acc_factor": acc_factor,
                  "K": K,
                  "n_x": n_x,
                  "n_y": n_y,
                  "x_grids": x_grids,
                  "y_grids": y_grids,
                  "train_model": train_model,
                  "pbar_update_interval": pbar_update_interval,
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
        if load_opt_dir != "":
            opt = joblib.load(load_opt_dir)
        else:
            opt = None
        for i, (num_iters, lr) in enumerate(zip(list_of_num_iters, list_of_lr)):
            losses, opt = model.fit(data, optimizer=opt, method='adam', num_iters=num_iters, lr=lr,
                                    masks=(masks_a, masks_b), pbar_update_interval=pbar_update_interval,
                                    memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
            list_of_losses.append(losses)

            checkpoint_dir = rslt_dir + "/checkpoint_{}".format(i)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                print("Creating checkpoint_{} directory...".format(i))
            # save model and opt
            joblib.dump(model, checkpoint_dir+"/model")
            joblib.dump(opt, checkpoint_dir+"/optimizer")
            # save rest
            rslt_saving(checkpoint_dir, model, Df, data, masks_a, masks_b, m_kwargs_a, m_kwargs_b, sample_T,
                        train_model, losses, quiver_scale)

    else:
        # only save the results
        rslt_saving(rslt_dir, model, Df, data, masks_a, masks_b, m_kwargs_a, m_kwargs_b, sample_T,
                    False, [], quiver_scale)

    print("Finish running!")


if __name__ == "__main__":
    main()
