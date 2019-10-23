from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import get_np

from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.coupled_transformations.gpgrid_transformation import GPGridTransformation, pairwise_xydist_sq
from project_ssms.feature_funcs import f_corner_vec_func
from project_ssms.momentum_utils import filter_traj_by_speed
from project_ssms.utils import downsample
from project_ssms.constants import ARENA_XMIN, ARENA_XMAX, ARENA_YMIN, ARENA_YMAX

from saver.rslts_saving import addDateTime, NumpyEncoder
from saver.runner_contgrid_rslt_saving import rslt_saving

import matplotlib.pyplot as plt

import torch
import numpy as np

import joblib
import git
import os
import click
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device {} \n\n".format(device))

################### specifying default arguments ################


@click.command()
@click.option('--job_name', default=None, help='name of the job')
@click.option('--downsample_n', default=1, help='downsample factor. Data size will reduce to 1/downsample_n')
@click.option('--filter_traj', is_flag=True, help='whether or not to filter the trajectory by SPEED')
@click.option('--use_log_prior', is_flag=True, help='whether to use log_prior to smooth the dynamics')
@click.option('--add_log_diagonal_prior', is_flag=True,
              help='whether to add log_diagonal_prior to smooth the dynamics diagonally')
@click.option('--log_prior_sigma_sq', default=-np.log(1e3), help='the variance for the weight smoothing prior')
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
@click.option('--held_out_proportion', default=0.05, help='the proportion of the held-out dataset in the whole dataset')
@click.option('--torch_seed', default=0, help='torch random seed')
@click.option('--np_seed', default=0, help='numpy random seed')
@click.option('--k', default=4, help='number of hidden states. Would be overwritten if load model.')
@click.option('--x_grids', default=None, help='x coordinates to specify the grids')
@click.option('--y_grids', default=None, help='y coordinates to specify the grids')
@click.option('--n_x', default=3, help='number of grids in x axis.'
                                       ' Would be overwritten if x_grids is provided, or load model.')
@click.option('--n_y', default=3, help='number of grids in y_axis.'
                                       ' Would be overwritten if y_grids is provided, or load model.')
@click.option('--rs_factor', default='30,30', help='rs factor. please input two numbers. '
                                                   'If want to use default, input 0,0')
@click.option('--rs', default=10, help='length scale')
@click.option('--train_rs', is_flag=True, help='whether to train length scale (r)')
@click.option('--list_of_num_iters', default='5000,5000',
              help='a list of checkpoint numbers of iterations for training')
@click.option('--list_of_lr', default='0.005, 0.005', help='learning rate for training')
@click.option('--ckpts_not_to_save', default=None, help='where to skip saving rslts')
@click.option('--list_of_k_steps', default='5', help='list of number of steps prediction forward')
@click.option('--sample_t', default=100, help='length of samples')
@click.option('--quiver_scale', default=0.8, help='scale for the quiver plots')
def main(job_name, downsample_n, filter_traj, use_log_prior, add_log_diagonal_prior, log_prior_sigma_sq,
         load_model, load_model_dir, load_opt_dir,
         transition, sticky_alpha, sticky_kappa, acc_factor, k, x_grids, y_grids, n_x, n_y, rs_factor, rs, train_rs,
         train_model, pbar_update_interval, video_clips, held_out_proportion, torch_seed, np_seed,
         list_of_num_iters, ckpts_not_to_save, list_of_lr, list_of_k_steps, sample_t, quiver_scale):
    if job_name is None:
        raise ValueError("Please provide the job name.")
    K = k
    sample_T = sample_t
    log_prior_sigma_sq = float(log_prior_sigma_sq)
    rs_factor = np.array([float(x) for x in rs_factor.split(",")])
    if rs_factor[0] == 0 and rs_factor[1] == 0:
        rs_factor = None
    rs = float(rs)
    video_clip_start, video_clip_end = [float(x) for x in video_clips.split(",")]
    list_of_num_iters = [int(x) for x in list_of_num_iters.split(",")]
    list_of_lr = [float(x) for x in list_of_lr.split(",")]
    list_of_k_steps = [int(x) for x in list_of_k_steps.split(",")]
    assert len(list_of_num_iters) == len(list_of_lr), "Length of list_of_num_iters must match length of list_of_lr."
    for lr in list_of_lr:
        if lr > 1:
            raise ValueError("Learning rate should not be larger than 1!")

    ckpts_not_to_save = ckpts_not_to_save if ckpts_not_to_save else ""
    ckpts_not_to_save = [int(x) for x in ckpts_not_to_save.split(',')]

    repo = git.Repo('.', search_parent_directories=True)  # SocialBehaviorectories=True)
    repo_dir = repo.working_tree_dir  # SocialBehavior

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)

    ########################## data ########################
    data_dir = repo_dir + '/SocialBehaviorptc/data/trajs_all'
    trajs = joblib.load(data_dir)

    traj = trajs[int(36000*video_clip_start):int(36000*video_clip_end)]
    traj = downsample(traj, downsample_n)
    if filter_traj:
        traj = filter_traj_by_speed(traj, q1=0.99, q2=0.99)

    data = torch.tensor(traj, dtype=torch.float64, device=device)
    assert 0 <= held_out_proportion < 0.2, \
        "held_out-portion should be between 0 and 0.2 (inclusive), but is {}".format(held_out_proportion)
    T = data.shape[0]
    breakpoint = int(T*(1-held_out_proportion))
    training_data = data[:breakpoint]
    valid_data = data[breakpoint:]

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

        acc_factor = tran.acc_factor

    else:
        print("Creating the model...")
        bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX],
                           [ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])

        # grids
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

        if acc_factor is None:
            acc_factor = downsample_n * 10

        tran = GPGridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids,
                                    Df=Df, feature_vec_func=f_corner_vec_func, acc_factor=acc_factor,
                                    rs_factor=rs_factor, rs=None, train_rs=train_rs,
                                    device=device)
        obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=1, bounds=bounds, transformation=tran, device=device)

        if transition == 'sticky':
            transition_kwargs = dict(alpha=sticky_alpha, kappa=sticky_kappa)
        else:
            transition_kwargs = None
        model = HMM(K=K, D=D, M=M, transition=transition, observation=obs, transition_kwargs=transition_kwargs,
                    device=device)
        model.observation.mus_init = training_data[0] * torch.ones(K, D, dtype=torch.float64, device=device)

    # save experiment params
    exp_params = {"job_name":   job_name,
                  'downsample_n': downsample_n,
                  "filter_traj": filter_traj,
                  "use_log_prior": use_log_prior,
                  "add_log_diagonal_prior": add_log_diagonal_prior,
                  "log_prior_sigma_sq": log_prior_sigma_sq,
                  "load_model": load_model,
                  "load_model_dir": load_model_dir,
                  "load_opt_dir": load_opt_dir,
                  "transition": transition,
                  "sticky_alpha": sticky_alpha,
                  "sticky_kappa": sticky_kappa,
                  "acc_factor": acc_factor,
                  "K": K,
                  "x_grids": x_grids,
                  "y_grids": y_grids,
                  "n_x": n_x,
                  "n_y": n_y,
                  "rs_factor": get_np(tran.rs_factor),
                  "rs": rs,
                  "train_rs": train_rs,
                  "train_model": train_model,
                  "pbar_update_interval": pbar_update_interval,
                  "video_clip_start": video_clip_start,
                  "video_clip_end": video_clip_end,
                  "held_out_proportion": held_out_proportion,
                  "torch_seed": torch_seed,
                  "np_seed": np_seed,
                  "list_of_num_iters": list_of_num_iters,
                  "list_of_lr": list_of_lr,
                  "list_of_k_steps": list_of_k_steps,
                  "sample_T": sample_T,
                  "quiver_scale": quiver_scale}

    print("Experiment params:")
    print(exp_params)

    rslt_dir = addDateTime("rslts/gp/" + job_name)
    rslt_dir = os.path.join(repo_dir, rslt_dir)
    if not os.path.exists(rslt_dir):
        os.makedirs(rslt_dir)
        print("Making result directory...")
    print("Saving to rlst_dir: ", rslt_dir)
    with open(rslt_dir+"/exp_params.json", "w") as f:
        json.dump(exp_params, f, indent=4, cls=NumpyEncoder)

    # compute memory
    print("Computing memory...")
    def get_memory_kwargs(data, train_rs):
        feature_vecs_a = f_corner_vec_func(data[:-1, 0:2])
        feature_vecs_b = f_corner_vec_func(data[:-1, 2:4])

        gpt_idx_a, grid_idx_a = tran.get_gpt_idx_and_grid_idx_for_batch(data[:-1, 0:2])
        gpt_idx_b, grid_idx_b = tran.get_gpt_idx_and_grid_idx_for_batch(data[:-1, 2:4])

        if train_rs:
            nearby_gpts_a = tran.gridpoints[gpt_idx_a]
            dist_sq_a = (data[:-1, None, 0:2] - nearby_gpts_a) ** 2

            nearby_gpts_b = tran.gridpoints[gpt_idx_b]
            dist_sq_b = (data[:-1, None, 2:4] - nearby_gpts_b) ** 2

            return dict(feature_vecs_a=feature_vecs_a, feature_vecs_b=feature_vecs_b,
                                 gpt_idx_a=gpt_idx_a, gpt_idx_b=gpt_idx_b, grid_idx_a=grid_idx_a,
                                 grid_idx_b=grid_idx_b, dist_sq_a=dist_sq_a, dist_sq_b=dist_sq_b)

        else:
            coeff_a = tran.get_gp_coefficients(data[:-1, 0:2], 0, gpt_idx_a, grid_idx_a)
            coeff_b = tran.get_gp_coefficients(data[:-1, 2:4], 0, gpt_idx_b, grid_idx_b)

            return dict(feature_vecs_a=feature_vecs_a, feature_vecs_b=feature_vecs_b,
                                 gpt_idx_a=gpt_idx_a, gpt_idx_b=gpt_idx_b, grid_idx_a=grid_idx_a,
                                 grid_idx_b=grid_idx_b, coeff_a=coeff_a, coeff_b=coeff_b)

    memory_kwargs = get_memory_kwargs(training_data, train_rs)
    valid_data_memory_kwargs = get_memory_kwargs(valid_data, train_rs)

    log_prob = model.log_likelihood(training_data, **memory_kwargs)

    ##################### training ############################
    if train_model:
        print("start training")
        list_of_losses = []
        if load_opt_dir != "":
            opt = joblib.load(load_opt_dir)
        else:
            opt = None
        for i, (num_iters, lr) in enumerate(zip(list_of_num_iters, list_of_lr)):
            training_losses, opt, valid_losses = model.fit(training_data, optimizer=opt, method='adam', num_iters=num_iters, lr=lr,
                                    pbar_update_interval=pbar_update_interval, valid_data=valid_data,
                                    valid_data_memory_kwargs=valid_data_memory_kwargs, **memory_kwargs)
            list_of_losses.append(training_losses)

            checkpoint_dir = rslt_dir + "/checkpoint_{}".format(i)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                print("Creating checkpoint_{} directory...".format(i))
            # save model and opt
            joblib.dump(model, checkpoint_dir+"/model")
            joblib.dump(opt, checkpoint_dir+"/optimizer")

            # save losses
            losses = dict(training_loss=training_losses, valid_loss=valid_losses)
            joblib.dump(losses, checkpoint_dir + "/losses")

            plt.figure()
            plt.plot(training_losses)
            plt.title("training loss")
            plt.savefig(rslt_dir + "/training_losses.jpg")
            plt.close()

            plt.figure()
            plt.plot(valid_losses)
            plt.title("validation loss")
            plt.savefig(rslt_dir + "/valid_losses.jpg")
            plt.close()

            # save rest
            if i in ckpts_not_to_save:
                print("ckpt {}: skip!\n".format(i))
                continue
            with torch.no_grad():
                rslt_saving(rslt_dir=checkpoint_dir, model=model, data=training_data, memory_kwargs=memory_kwargs,
                            list_of_k_steps=list_of_k_steps, sample_T=sample_T,quiver_scale=quiver_scale,
                            valid_data=valid_data,  valid_data_memory_kwargs=valid_data_memory_kwargs, device=device)

    else:
        # only save the results
        rslt_saving(rslt_dir=rslt_dir, model=model, data=training_data, memory_kwargs=memory_kwargs,
                    list_of_k_steps=list_of_k_steps, sample_T=sample_T, quiver_scale=quiver_scale,
                    valid_data=valid_data, valid_data_memory_kwargs=valid_data_memory_kwargs, device=device)

    print("Finish running!")


if __name__ == "__main__":
    main()

# --train_model --job_name=local/v35_downsamplen2_K6 --video_clips=3,5 --downsample_n=2 --k=6 --n_x=6 --n_y=6
# --list_of_num_iters=1000,2000,3000,5000,2000,2000 --list_of_lr=0.5,0.1,0.05,0.01,0.005,0.005
# --sample_t=36000 --pbar_update_interval=10

