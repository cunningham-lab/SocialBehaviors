from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import get_np

from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.coupled_transformations.lineargrid_transformation import LinearGridTransformation
from project_ssms.single_transformations.single_lineargird_transformation import SingleLinearGridTransformation
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

################### specifying default arguments ################


@click.command()
@click.option('--job_name', default=None, help='name of the job')
@click.option('--cuda_num', default=0, help='which cuda device to use')
@click.option('--data_type', default='full', help='choose from full, selected_010_virgin')
@click.option('--downsample_n', default=1, help='downsample factor. Data size will reduce to 1/downsample_n')
@click.option('--filter_traj', is_flag=True, help='whether or not to filter the trajectory by SPEED')
@click.option('--use_log_prior', is_flag=True, help='whether to use log_prior to smooth the dynamics')
@click.option('--lg_version', default=1, help='version of the lp grid')
@click.option('--no_boundary_prior', is_flag=True, help='whether to drop priors on the boundary')
@click.option('--add_log_diagonal_prior', is_flag=True,
              help='whether to add log_diagonal_prior to smooth the dynamics diagonally')
@click.option('--log_prior_sigma_sq', default=-np.log(1e3), help='the variance for the weight smoothing prior')
@click.option('--load_model', is_flag=True, help='Whether to load the (trained) model')
@click.option('--load_model_dir', default="", help='Directory of model to load')
@click.option('--reset_prior_info', is_flag=True, help='wheter to reset prior info when loading the model')
@click.option('--transition', default="stationary", help='type of transition (str)')
@click.option('--sticky_alpha', default=1, help='value of alpha in sticky transition')
@click.option('--sticky_kappa', default=100, help='value of kappa in sticky transition')
@click.option('--acc_factor', default=None, help="acc factor in direction model")
@click.option('--train_model', is_flag=True, help='Whether to train the model')
@click.option('--pbar_update_interval', default=500, help='progress bar update interval')
@click.option('--load_opt_dir', default="", help='Directory of optimizer to load.')
@click.option('--animal', default="both", help='choose among both, virgin and mother')
@click.option('--prop_start_end', default='0,1.0', help='starting and ending proportion. '
                                                        'useful when data_type is selected')
@click.option('--video_clips', default="0,1", help='The starting video clip of the training data. '
                                                   'useful when data_tye is full')
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
@click.option('--list_of_num_iters', default='5000,5000',
              help='a list of checkpoint numbers of iterations for training')
@click.option('--ckpts_not_to_save', default=None, help='where to skip saving rslts')
@click.option('--list_of_lr', default='0.005, 0.005', help='learning rate for training')
@click.option('--list_of_k_steps', default=None, help='list of number of steps prediction forward')
@click.option('--sample_t', default=100, help='length of samples')
@click.option('--quiver_scale', default=0.8, help='scale for the quiver plots')
def main(job_name, cuda_num, data_type, downsample_n, filter_traj, lg_version, use_log_prior, no_boundary_prior,
         add_log_diagonal_prior, log_prior_sigma_sq,
         load_model, load_model_dir, load_opt_dir, animal, reset_prior_info,
         transition, sticky_alpha, sticky_kappa, acc_factor, k, x_grids, y_grids, n_x, n_y,
         train_model, pbar_update_interval, prop_start_end, video_clips, held_out_proportion, torch_seed, np_seed,
         list_of_num_iters, ckpts_not_to_save, list_of_lr, list_of_k_steps, sample_t, quiver_scale):
    if job_name is None:
        raise ValueError("Please provide the job name.")
    assert animal in ['both', 'virgin', 'mother'], animal

    cuda_num = int(cuda_num)
    device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")
    print("Using device {} \n\n".format(device))

    K = k
    sample_T = sample_t
    video_clip_start, video_clip_end = [float(x) for x in video_clips.split(",")]
    start, end = [float(x) for x in prop_start_end.split(",")]
    log_prior_sigma_sq = float(log_prior_sigma_sq)
    list_of_num_iters = [int(x) for x in list_of_num_iters.split(",")]
    list_of_lr = [float(x) for x in list_of_lr.split(",")]
    # TODO: test if that works
    list_of_k_steps = [int(x) for x in list_of_k_steps.split(",")] if list_of_k_steps else []
    assert len(list_of_num_iters) == len(list_of_lr), "Length of list_of_num_iters must match length of list_of_lr."
    for lr in list_of_lr:
        if lr > 1:
            raise ValueError("Learning rate should not be larger than 1!")

    ckpts_not_to_save = [int(x) for x in ckpts_not_to_save.split(',')] if ckpts_not_to_save else []

    repo = git.Repo('.', search_parent_directories=True)  # SocialBehaviorectories=True)
    repo_dir = repo.working_tree_dir  # SocialBehavior

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)

    ########################## data ########################
    if data_type == 'full':
        data_dir = repo_dir + '/SocialBehaviorptc/data/trajs_all'
        traj = joblib.load(data_dir)
        traj = traj[int(36000 * video_clip_start):int(36000 * video_clip_end)]
    elif data_type == 'selected_010_virgin':
        assert animal == 'virgin', "animal much be 'virgin', but got {}.".format(animal)
        data_dir = repo_dir + '/SocialBehaviorptc/data/traj_010_virgin_selected'
        traj = joblib.load(data_dir)
        T = len(traj)
        traj = traj[int(T*start): int(T*end)]
    else:
        raise ValueError("unsupported data type: {}".format(data_type))

    if animal == 'virgin':
        traj = traj[:,0:2]
    elif animal == 'mother':
        traj = traj[:,2:4]

    traj = downsample(traj, downsample_n)
    if filter_traj:
        traj = filter_traj_by_speed(traj, q1=0.99, q2=0.99)

    data = torch.tensor(traj, dtype=torch.float64, device=device)
    assert 0 <= held_out_proportion <= 0.4, \
        "held_out-portion should be between 0 and 0.4 (inclusive), but is {}".format(held_out_proportion)
    T = data.shape[0]
    breakpoint = int(T*(1-held_out_proportion))
    training_data = data[:breakpoint]
    valid_data = data[breakpoint:]

    sample_T = training_data.shape[0]

    ######################### model ####################

    # model
    D = data.shape[1]
    assert D == 2 or 4, D
    M = 0
    Df = 4

    if load_model:
        print("Loading the model from ", load_model_dir)
        if reset_prior_info:

            pretrained_model = joblib.load(load_model_dir)
            pretrained_transition = pretrained_model.transition
            pretrained_observation = pretrained_model.observation
            pretrained_tran = pretrained_model.observation.transformation

            # set prior info
            pretrained_tran.use_log_prior = use_log_prior
            pretrained_tran.no_boundary_prior = no_boundary_prior
            pretrained_tran.add_log_diagonal_prior = add_log_diagonal_prior
            pretrained_tran.log_prior_sigma_sq = torch.tensor(log_prior_sigma_sq, dtype=torch.float64, device=device)

            acc_factor = pretrained_tran.acc_factor

            K = pretrained_model.K

            obs = ARTruncatedNormalObservation(K=K, D=D, M=0, obs=pretrained_observation, device=device)
            tran = obs.transformation

            if transition == 'sticky':
                transition_kwargs = dict(alpha=sticky_alpha, kappa=sticky_kappa)
            else:
                transition_kwargs = None
            model = HMM(K=K, D=D, M=M, pi0=get_np(pretrained_model.pi0), Pi=get_np(pretrained_transition.Pi),
                        transition=transition, observation=obs, transition_kwargs=transition_kwargs,
                        device=device)
            model.observation.mus_init = training_data[0] * torch.ones(K, D, dtype=torch.float64, device=device)
        else:
            model = joblib.load(load_model_dir)
            tran = model.observation.transformation

            K = model.K

            n_x = len(tran.x_grids) - 1
            n_y = len(tran.y_grids) - 1
            if isinstance(model, LinearGridTransformation):
                acc_factor = tran.acc_factor
                use_log_prior = tran.use_log_prior
                no_boundary_prior = tran.no_boundary_prior
                add_log_diagonal_prior = tran.add_log_diagonal_prior
                log_prior_sigma_sq = get_np(tran.log_prior_sigma_sq)

    else:
        if D == 4:
            bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX],
                               [ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])
        else:
            bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])

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

        print("Creating the model...")
        if animal == 'both':
            tran = LinearGridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids,
                                            Df=Df, feature_vec_func=f_corner_vec_func, acc_factor=acc_factor,
                                            use_log_prior=use_log_prior, no_boundary_prior=no_boundary_prior,
                                            add_log_diagonal_prior=add_log_diagonal_prior,
                                            log_prior_sigma_sq=log_prior_sigma_sq, device=device, version=lg_version)
        else:
            tran = SingleLinearGridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, device=device)
        obs = ARTruncatedNormalObservation(K=K, D=D, M=M, lags=1, bounds=bounds, transformation=tran, device=device)

        if transition == 'sticky':
            transition_kwargs = dict(alpha=sticky_alpha, kappa=sticky_kappa)
        elif transition == 'grid':
            transition_kwargs = dict(x_grids=x_grids, y_grids=y_grids)
        else:
            transition_kwargs = None
        model = HMM(K=K, D=D, M=M, transition=transition, observation=obs, transition_kwargs=transition_kwargs,
                    device=device)
        model.observation.mus_init = training_data[0] * torch.ones(K, D, dtype=torch.float64, device=device)

    # save experiment params
    exp_params = {"job_name":   job_name,
                  'downsample_n': downsample_n,
                  'data_type': data_type,
                  "filter_traj": filter_traj,
                  "lg_version": lg_version,
                  "use_log_prior": use_log_prior,
                  "add_log_diagonal_prior": add_log_diagonal_prior,
                  "no_boundary_prior": no_boundary_prior,
                  "log_prior_sigma_sq": log_prior_sigma_sq,
                  "load_model": load_model,
                  "load_model_dir": load_model_dir,
                  "animal": animal,
                  "reset_prior_info": reset_prior_info,
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
                  "train_model": train_model,
                  "pbar_update_interval": pbar_update_interval,
                  "video_clip_start": video_clip_start,
                  "video_clip_end": video_clip_end,
                  "start_percentage": start,
                  "end_percentage": end,
                  "held_out_proportion": held_out_proportion,
                  "torch_seed": torch_seed,
                  "np_seed": np_seed,
                  "list_of_num_iters": list_of_num_iters,
                  "ckpts_not_to_save": ckpts_not_to_save,
                  "list_of_lr": list_of_lr,
                  "list_of_k_steps": list_of_k_steps,
                  "sample_T": sample_T,
                  "quiver_scale": quiver_scale}

    print("Experiment params:")
    print(exp_params)

    rslt_dir = addDateTime("rslts/lineargrid/{}/{}".format(animal, job_name))
    rslt_dir = os.path.join(repo_dir, rslt_dir)
    if not os.path.exists(rslt_dir):
        os.makedirs(rslt_dir)
        print("Making result directory...")
    print("Saving to rlst_dir: ", rslt_dir)
    with open(rslt_dir+"/exp_params.json", "w") as f:
        json.dump(exp_params, f, indent=4, cls=NumpyEncoder)

    # compute memory
    if transition == "grid":
        print("Computing transition memory...")
        joint_grid_idx = model.transition.get_grid_idx(training_data[:-1])
        transition_memory_kwargs = dict(joint_grid_idx=joint_grid_idx)
        valid_joint_grid_idx = model.transition.get_grid_idx(valid_data[:-1])
        valid_data_transition_memory_kwargs = dict(joint_grid_idx=valid_joint_grid_idx)
    else:
        transition_memory_kwargs = None
        valid_data_transition_memory_kwargs = None

    print("Computing observation memory...")
    if animal == 'both':
        gridpoints_idx_a = tran.get_gridpoints_idx_for_batch(training_data[:-1, 0:2])  # (T-1, n_gps, 4)
        gridpoints_idx_b = tran.get_gridpoints_idx_for_batch(training_data[:-1, 2:4])  # (T-1, n_gps, 4)
        gridpoints_a = tran.get_gridpoints_for_batch(gridpoints_idx_a)  # (T-1, d, 2)
        gridpoints_b = tran.get_gridpoints_for_batch(gridpoints_idx_b)  # (T-1, d, 2)
        feature_vecs_a = f_corner_vec_func(training_data[:-1, 0:2])  # (T, Df, 2)
        feature_vecs_b = f_corner_vec_func(training_data[:-1, 2:4])  # (T, Df, 2)

        gridpoints_idx = (gridpoints_idx_a, gridpoints_idx_b)
        gridpoints = (gridpoints_a, gridpoints_b)
        feature_vecs = (feature_vecs_a, feature_vecs_b)
        memory_kwargs = dict(gridpoints_idx=gridpoints_idx, gridpoints=gridpoints, feature_vecs=feature_vecs)

        if len(valid_data) > 0:
            gridpoints_idx_a_v = tran.get_gridpoints_idx_for_batch(valid_data[:-1, 0:2])  # (T-1, n_gps, 4)
            gridpoints_idx_b_v = tran.get_gridpoints_idx_for_batch(valid_data[:-1, 2:4])  # (T-1, n_gps, 4)
            gridpoints_a_v = tran.get_gridpoints_for_batch(gridpoints_idx_a_v)  # (T-1, d, 2)
            gridpoints_b_v = tran.get_gridpoints_for_batch(gridpoints_idx_b_v)  # (T-1, d, 2)
            feature_vecs_a_v = f_corner_vec_func(valid_data[:-1, 0:2])  # (T, Df, 2)
            feature_vecs_b_v = f_corner_vec_func(valid_data[:-1, 2:4])  # (T, Df, 2)

            gridpoints_idx_v = (gridpoints_idx_a_v, gridpoints_idx_b_v)
            gridpoints_v = (gridpoints_a_v, gridpoints_b_v)
            feature_vecs_v = (feature_vecs_a_v, feature_vecs_b_v)
            valid_data_memory_kwargs = dict(gridpoints_idx=gridpoints_idx_v, gridpoints=gridpoints_v,
                                            feature_vecs=feature_vecs_v)
        else:
            valid_data_memory_kwargs = {}
    else:
        def get_memory_kwargs(data):
            if data is None or data.shape[0] == 0:
                return {}

            gridpoints_idx = tran.get_gridpoints_idx_for_batch(data)
            gridpoints = tran.gridpoints[gridpoints_idx]
            coeffs = tran.get_lp_coefficients(data, gridpoints[:, 0], gridpoints[:, 3], device=device)

            return dict(gridpoints_idx=gridpoints_idx, coeffs=coeffs)

        memory_kwargs = get_memory_kwargs(training_data[:-1])
        valid_data_memory_kwargs = get_memory_kwargs(valid_data[:-1])

    log_prob = model.log_likelihood(training_data, transition_memory_kwargs=transition_memory_kwargs, **memory_kwargs)
    print("log_prob = {}".format(log_prob))

    ##################### training ############################
    if train_model:
        print("start training")
        list_of_losses = []
        if load_opt_dir != "":
            opt = joblib.load(load_opt_dir)
        else:
            opt = None
        for i, (num_iters, lr) in enumerate(zip(list_of_num_iters, list_of_lr)):
            training_losses, opt, valid_losses, _ = \
                model.fit(training_data, optimizer=opt, method='adam', num_iters=num_iters, lr=lr,
                          pbar_update_interval=pbar_update_interval, valid_data=valid_data,
                          transition_memory_kwargs=transition_memory_kwargs,
                          valid_data_transition_memory_kwargs=valid_data_transition_memory_kwargs,
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
            joblib.dump(losses, checkpoint_dir+"/losses")

            plt.figure()
            plt.plot(training_losses)
            plt.title("training loss")
            plt.savefig(checkpoint_dir + "/training_losses.jpg")
            plt.close()

            plt.figure()
            plt.plot(valid_losses)
            plt.title("validation loss")
            plt.savefig(checkpoint_dir + "/valid_losses.jpg")
            plt.close()
            # save rest
            if i in ckpts_not_to_save:
                print("ckpt {}: skip!\n".format(i))
                continue
            with torch.no_grad():
                rslt_saving(rslt_dir=checkpoint_dir, model=model, data=training_data, animal=animal,
                            memory_kwargs=memory_kwargs,
                            list_of_k_steps=list_of_k_steps, sample_T=sample_T,
                            quiver_scale=quiver_scale, valid_data=valid_data,
                            valid_data_memory_kwargs=valid_data_memory_kwargs, device=device)

    else:
        # only save the results
        rslt_saving(rslt_dir=rslt_dir, model=model, data=training_data, animal=animal,
                    memory_kwargs=memory_kwargs,
                    list_of_k_steps=list_of_k_steps, sample_T=sample_T,
                    quiver_scale=quiver_scale, valid_data=valid_data,
                    valid_data_memory_kwargs=valid_data_memory_kwargs, device=device)

    print("Finish running!")


if __name__ == "__main__":
    main()

# --train_model --downsample_n=2 --job_name=local/test_general --video_clips=0,1 --transition=stationary
# --n_x=4 --n_y=4 --list_of_num_iters=50 --list_of_lr=0.005 --list_of_k_steps=5,10 --sample_t=100
# --pbar_update_interval=10

# --train_model --downsample_n=2 --job_name=local/train_v01 --video_clips=0,1 --transition=stationary
# --n_x=4 --n_y=4 --list_of_num_iters=3000,2000 --list_of_lr=0.1,0.05 --sample_t=10000 --pbar_update_interval=10

