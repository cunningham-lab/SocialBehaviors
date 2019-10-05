from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.truncated_normal_observation import TruncatedNormalObservation

from project_ssms.momentum_utils import filter_traj_by_speed
from project_ssms.utils import downsample
from project_ssms.constants import ARENA_XMIN, ARENA_XMAX, ARENA_YMIN, ARENA_YMAX

from saver.rslts_saving import addDateTime, NumpyEncoder
from saver.runner_hmm_rslt_saving import rslt_saving

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
@click.option('--mouse', default='both', help='choose from virgin, mother or both')
@click.option('--filter_traj', is_flag=True, help='whether or not to filter the trajectory by SPEED')
@click.option('--load_model', is_flag=True, help='Whether to load the (trained) model')
@click.option('--load_model_dir', default="", help='Directory of model to load')
@click.option('--transition', default="stationary", help='type of transition (str)')
@click.option('--sticky_alpha', default=1, help='value of alpha in sticky transition')
@click.option('--sticky_kappa', default=100, help='value of kappa in sticky transition')
@click.option('--not_train_mu', is_flag=True, help='whether to train mu')
@click.option('--initialize_mu', is_flag=True, help='whether to initialize mu')
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
@click.option('--quiver_scale', default=0.8, help='scale for the quiver plots')
def main(job_name, downsample_n, mouse, filter_traj, load_model, load_model_dir, load_opt_dir,
         transition, sticky_alpha, sticky_kappa, k, x_grids, y_grids, n_x, n_y, not_train_mu, initialize_mu,
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

    # specify grids for sanity checks and plotting
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

    ########################## data ########################
    data_dir = repo_dir + '/SocialBehaviorptc/data/trajs_all'
    trajs = joblib.load(data_dir)

    traj = trajs[36000*video_clip_start:36000*video_clip_end]
    traj = downsample(traj, downsample_n)
    if filter_traj:
        traj = filter_traj_by_speed(traj, q1=0.99, q2=0.99)

    if mouse == "both":
        data = torch.tensor(traj, dtype=torch.float64)
        D = 4
    elif mouse == "virgin":
        data = torch.tensor(traj[:, 0:2], dtype=torch.float64)
        D = 2
    elif mouse == "mother":
        data = torch.tensor(traj[:, 2:4], dtype=torch.float64)
        D = 2
    else:
        raise ValueError("mouse must be chosen from 'both', 'virgin', 'mother'.")

    ######################### model ####################

    # model
    M = 0

    if load_model:
        print("Loading the model from ", load_model_dir)
        model = joblib.load(load_model_dir)

        K = model.K
        if model.observation.mus.requires_grad:
            cluster_locations = model.observation.mus.detach().numpy()
        else:
            cluster_locations = model.observation.mus.numpy()
    else:
        print("Creating the model...")
        if D == 4:
            bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX],
                               [ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])
        else:
            bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])
        
        if transition == 'sticky':
            transition_kwargs = dict(alpha=sticky_alpha, kappa=sticky_kappa)
        else:
            transition_kwargs = None
            
        obs = TruncatedNormalObservation(K=K, D=D, M=M, bounds=bounds)
        model = HMM(K=K, D=D, M=M, transition=transition, observation=obs, transition_kwargs=transition_kwargs)

        if initialize_mu:
            if mouse == 'both':
                assert K == (n_x * n_y) ** 2, "K should be equal to (n_x*n_y)**2"
            else:
                assert K == n_x * n_y, "K should be equal to n_x * n_y"

            cluster_locations = np.array(
                [((x_grids[i] + x_grids[i + 1]) / 2, (y_grids[j] + y_grids[j + 1]) / 2) for i in range(n_x) for j in
                 range(n_y)])
            if mouse == "both":
                cluster_locations = np.array([np.concatenate((loc_a, loc_b))
                                           for loc_a in cluster_locations for loc_b in cluster_locations])

            assert cluster_locations.shape == (K, D)
            train_mu = not not_train_mu
            model.observation.mus = torch.tensor(cluster_locations, dtype=torch.float64, requires_grad=train_mu)
        else:
            if model.observation.mus.requires_grad:
                cluster_locations = model.observation.mus.detach().numpy()
            else:
                cluster_locations = model.observation.mus.numpy()
            
    # save experiment params
    exp_params = {"job_name":   job_name,
                  'downsample_n': downsample_n,
                  'mouse': mouse,
                  "load_model": load_model,
                  "load_model_dir": load_model_dir,
                  "load_opt_dir": load_opt_dir,
                  "transition": transition,
                  "sticky_alpha": sticky_alpha,
                  "sticky_kappa": sticky_kappa,
                  "K": K,
                  "n_x": n_x,
                  "n_y": n_y,
                  "x_grids": x_grids,
                  "y_grids": y_grids,
                  "cluster_locations": cluster_locations,
                  "not_train_mu": not_train_mu,
                  "initialize_mu": initialize_mu,
                  "train_model": train_model,
                  "pbar_update_interval": pbar_update_interval,
                  "list_of_num_iters": list_of_num_iters,
                  "list_of_lr": list_of_lr,
                  "video_clip_start": video_clip_start,
                  "video_clip_end": video_clip_end,
                  "sample_T": sample_T}

    print("Experiment params:")
    print(exp_params)

    rslt_dir = addDateTime("rslts/hmm/" + job_name)
    rslt_dir = os.path.join(repo_dir, rslt_dir)
    if not os.path.exists(rslt_dir):
        os.makedirs(rslt_dir)
        print("Making result directory...")
    print("Saving to rlst_dir: ", rslt_dir)
    with open(rslt_dir+"/exp_params.json", "w") as f:
        json.dump(exp_params, f, indent=4, cls=NumpyEncoder)

    ##################### training ############################
    if train_model:
        initial_dir = rslt_dir + "/initial"
        if not os.path.exists(initial_dir):
            os.makedirs(initial_dir)
            print("\nCreating initial directory...")
        rslt_saving(initial_dir, model, data, mouse, sample_T,
                    False, [], quiver_scale, x_grids, y_grids)

        print("start training")
        list_of_losses = []
        if load_opt_dir != "":
            opt = joblib.load(load_opt_dir)
        else:
            opt = None
        for i, (num_iters, lr) in enumerate(zip(list_of_num_iters, list_of_lr)):
            losses, opt = model.fit(data, optimizer=opt, method='adam', num_iters=num_iters, lr=lr,
                                    pbar_update_interval=pbar_update_interval)
            list_of_losses.append(losses)

            checkpoint_dir = rslt_dir + "/checkpoint_{}".format(i)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                print("\nCreating checkpoint_{} directory...".format(i))
            # save model and opt
            joblib.dump(model, checkpoint_dir+"/model")
            joblib.dump(opt, checkpoint_dir+"/optimizer")
            # save rest
            rslt_saving(checkpoint_dir, model, data, mouse, sample_T,
                        train_model, losses, quiver_scale, x_grids, y_grids)

    else:
        # only save the results
        rslt_saving(rslt_dir, model, data, mouse, sample_T,
                    False, [], quiver_scale, x_grids, y_grids)

    print("Finish running!")


if __name__ == "__main__":
    main()

# --train_model --downsample_n=2 --job_name=local/test_general --video_clips=0,1 --k=20 --n_x=6 --n_y=6
# --list_of_num_iters=10,20 --list_of_lr=0.1,0.05 --sample_t=10 --pbar_update_interval=10

# --train_model --downsample_n=2 --job_name=local/train_v01 --video_clips=0,1 --transition=stationary
# --n_x=4 --n_y=4 --list_of_num_iters=3000,2000 --list_of_lr=0.1,0.05 --sample_t=10000 --pbar_update_interval=10

### single animal --virgin
# --job_name=single_virgin/ --train_model --mouse=virgin --downsample_n=2 --k=4 --n_x=2 --n_y=2
# --list_of_num_iters=1000,1000,1000,1000,1000 --list_of_lr=0.5,0.5,0.5,0.5,0.5 --sample_t=18000 --pbar_update_interval=10

### single_animal --mother
# --job_name=single_mother/ --train_model --mouse=mother --downsample_n=2 --k=4 --n_x=2 --n_y=2
# --list_of_num_iters=1000,1000,1000,1000,1000 --list_of_lr=0.5,0.5,0.5,0.5,0.5 --sample_t=18000 --pbar_update_interval=10

### both animals
# --job_name=both/ --train_model --mouse=both --downsample_n=2 --k=16 --n_x=2 --n_y=2
# --list_of_num_iters=1000,1000,1000,1000,1000 --list_of_lr=0.5,0.5,0.5,0.5,0.5 --sample_t=18000 --pbar_update_interval=10

