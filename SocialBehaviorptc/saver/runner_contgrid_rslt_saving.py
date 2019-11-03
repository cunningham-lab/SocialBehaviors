import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib

from project_ssms.coupled_transformations.lineargrid_transformation import LinearGridTransformation
from project_ssms.coupled_transformations.gpgrid_transformation import GPGridTransformation
from project_ssms.coupled_transformations.weightedgrid_transformation import WeightedGridTransformation
from project_ssms.coupled_transformations.lstm_transformation import LSTMTransformation
from project_ssms.coupled_transformations.uni_lstm_transformation import UniLSTMTransformation
from project_ssms.coupled_transformations.lstm_based_transformation import LSTMBasedTransformation
from project_ssms.utils import k_step_prediction_for_lineargrid_model, k_step_prediction_for_gpgrid_model, \
    k_step_prediction_for_weightedgrid_model, \
    k_step_prediction_for_lstm_model, k_step_prediction_for_lstm_based_model
from project_ssms.plot_utils import plot_z, plot_mouse, plot_data_condition_on_all_zs, plot_2d_time_plot_condition_on_all_zs
from project_ssms.grid_utils import plot_quiver, plot_realdata_quiver, \
    get_all_angles, get_speed, plot_list_of_angles, plot_list_of_speed, plot_space_dist
from project_ssms.constants import *

from ssm_ptc.utils import k_step_prediction, get_np

from saver.rslts_saving import NumpyEncoder


def rslt_saving(rslt_dir, model, data, memory_kwargs, list_of_k_steps, sample_T,
                quiver_scale, x_grids=None, y_grids=None, dynamics_T=None,
                valid_data=None, valid_data_memory_kwargs=None, device=torch.device('cpu')):

    valid_data_memory_kwargs = valid_data_memory_kwargs if valid_data_memory_kwargs else {}

    tran = model.observation.transformation
    if x_grids is None or y_grids is None:
        x_grids = tran.x_grids
        y_grids = tran.y_grids
    n_x = len(x_grids) - 1
    n_y = len(y_grids) - 1

    K = model.K

    memory_kwargs = memory_kwargs if memory_kwargs else {}


    #################### inference ###########################

    print("\ninferring most likely states...")
    z = model.most_likely_states(data, **memory_kwargs)
    z_valid = model.most_likely_states(valid_data, **valid_data_memory_kwargs)

    # TODO: address valida_data = None
    print("0 step prediction")
    # TODO: add valid data for other model
    if data.shape[0] <= 10000:
        data_to_predict = data
    else:
        data_to_predict = data[-10000:]
    if isinstance(tran, LinearGridTransformation):
        x_predict = k_step_prediction_for_lineargrid_model(model, z, data_to_predict, **memory_kwargs)
        x_predict_valid = k_step_prediction_for_lineargrid_model(model, z_valid, valid_data, **valid_data_memory_kwargs)
    elif isinstance(tran, GPGridTransformation):
        x_predict = k_step_prediction_for_gpgrid_model(model, z, data_to_predict, **memory_kwargs)
        x_predict_valid = k_step_prediction_for_gpgrid_model(model, z_valid, valid_data, **valid_data_memory_kwargs)
    elif isinstance(tran, WeightedGridTransformation):
        x_predict = k_step_prediction_for_weightedgrid_model(model, z, data_to_predict, **memory_kwargs)
        x_predict_valid = k_step_prediction_for_weightedgrid_model(model, z_valid, valid_data, **valid_data_memory_kwargs)
    elif isinstance(tran, (LSTMTransformation, UniLSTMTransformation)):
        x_predict = k_step_prediction_for_lstm_model(model, z, data_to_predict,
                                                     feature_vecs=memory_kwargs["feature_vecs"])
        x_predict_valid = k_step_prediction_for_lstm_model(model, z_valid, valid_data,
                                                           feature_vecs=valid_data_memory_kwargs["feature_vecs"])
    elif isinstance(tran, LSTMBasedTransformation):
        x_predict = k_step_prediction_for_lstm_based_model(model, z, data_to_predict, k=0,
                                                           feature_vecs=memory_kwargs["feature_vecs"])
        x_predict_valid = k_step_prediction_for_lstm_based_model(model, z_valid, valid_data, k=0,
                                                           feature_vecs=valid_data_memory_kwargs["feature_vecs"])
    else:
        raise ValueError("Unsupported transformation!")
    x_predict_err = np.mean(np.abs(x_predict - get_np(data_to_predict)), axis=0)
    if len(valid_data) == 0:
        x_predict_valid_err = None
    else:
        x_predict_valid_err = np.mean(np.abs(x_predict_valid - get_np(valid_data)), axis=0)

    dict_of_x_predict_k = dict(x_predict_0=x_predict, x_predict_v_0=x_predict_valid)
    dict_of_x_predict_k_err = dict(x_predict_0_err=x_predict_err, x_predict_v_0_err=x_predict_valid_err)

    for k_step in list_of_k_steps:
        print("{} step prediction".format(k_step))
        if isinstance(tran, LSTMBasedTransformation):
            # TODO: take care of empty valid data
            x_predict_k = k_step_prediction_for_lstm_based_model(model, z, data_to_predict, k=k_step)
            x_predict_valid_k = k_step_prediction_for_lstm_model(model, z, data_to_predict)
        else:
            x_predict_k = k_step_prediction(model, z, data_to_predict, k=k_step)
            x_predict_valid_k = k_step_prediction(model, z_valid, valid_data, k=k_step)
        x_predict_k_err = np.mean(np.abs(x_predict_k - get_np(data_to_predict[k_step:])), axis=0)
        if len(valid_data) == 0:
            x_predict_valid_k_err = None
        else:
            x_predict_valid_k_err = np.mean(np.abs(x_predict_valid_k - get_np(valid_data[k_step:])), axis=0)
        dict_of_x_predict_k["x_predict_{}".format(k_step)] = x_predict_k
        dict_of_x_predict_k["x_predict_v_{}".format(k_step)] = x_predict_valid_k
        dict_of_x_predict_k_err["x_predict_{}_err".format(k_step)] = x_predict_k_err
        dict_of_x_predict_k_err["x_predict_v_{}_err".format(k_step)] = x_predict_valid_k_err


    ################### samples #########################
    print("sampling")
    center_z = torch.tensor([0], dtype=torch.int, device=device)
    center_x = torch.tensor([[150, 190, 200, 200]], dtype=torch.float64, device=device)

    if isinstance(tran, LSTMBasedTransformation):
        lstm_states = {}
        sample_z, sample_x = model.sample(sample_T, lstm_states=lstm_states)

        lstm_states = {}
        sample_z_center, sample_x_center = model.sample(sample_T, prefix=(center_z, center_x), lstm_states=lstm_states)
    else:
        sample_z, sample_x = model.sample(sample_T)

        sample_z_center, sample_x_center = model.sample(sample_T, prefix=(center_z, center_x))


    ################## dynamics #####################

    if isinstance(tran, (LinearGridTransformation, GPGridTransformation, WeightedGridTransformation)):
        # quiver
        XX, YY = np.meshgrid(np.linspace(20, 310, 30),
                             np.linspace(0, 380, 30))
        XY = np.column_stack((np.ravel(XX), np.ravel(YY)))  # shape (900,2) grid values
        XY_grids = np.concatenate((XY, XY), axis=1)  # (900, 4)

        XY_next = tran.transform(torch.tensor(XY_grids, dtype=torch.float64, device=device))
        dXY = get_np(XY_next) - XY_grids[:, None]

    # TODO: maybe use sample condition on z (transformation) to show the dynamics
    samples_on_fixed_zs = []
    if isinstance(tran, LSTMBasedTransformation):
        assert dynamics_T is not None
        for k in range(K):
            lstm_states = {}
            fixed_z = torch.ones(dynamics_T, dtype=torch.int) * k
            samples_on_fixed_z = model.sample_condition_on_zs(zs=fixed_z, transformation=True, return_np=True,
                                                              lstm_states=lstm_states)
            samples_on_fixed_zs.append(samples_on_fixed_z)

    #################### saving ##############################

    print("begin saving...")

    # save summary
    if isinstance(tran, (LinearGridTransformation, GPGridTransformation, WeightedGridTransformation)):
        avg_transform_speed = np.average(np.abs(dXY), axis=0)
    avg_sample_speed = np.average(np.abs(np.diff(sample_x, axis=0)), axis=0)
    avg_sample_center_speed = np.average(np.abs(np.diff(sample_x_center, axis=0)), axis=0)
    avg_data_speed = np.average(np.abs(np.diff(get_np(data), axis=0)), axis=0)

    transition_matrix = model.transition.stationary_transition_matrix
    if transition_matrix.requires_grad:
        transition_matrix = get_np(transition_matrix)
    else:
        transition_matrix = get_np(transition_matrix)
    summary_dict = {"init_dist": get_np(model.init_dist),
                    "transition_matrix": transition_matrix,
                    "variance": get_np(torch.exp(model.observation.log_sigmas)),
                    "log_likes": get_np(model.log_likelihood(data, **memory_kwargs)),
                    "avg_data_speed": avg_data_speed,
                    "avg_sample_speed": avg_sample_speed, "avg_sample_center_speed": avg_sample_center_speed}
    summary_dict = {**dict_of_x_predict_k_err, **summary_dict}
    if len(valid_data) > 0:
        summary_dict["valid_log_likes"] = get_np(model.log_likelihood(valid_data, **valid_data_memory_kwargs))
    if isinstance(tran, GPGridTransformation):
        summary_dict["real_rs"] = get_np(tran.rs_factor * torch.sigmoid(tran.rs))
    if isinstance(tran, WeightedGridTransformation):
        summary_dict["beta"] = get_np(tran.beta)
    if isinstance(tran, (LinearGridTransformation, GPGridTransformation, WeightedGridTransformation)):
        summary_dict["avg_transform_speed"] = avg_transform_speed
    with open(rslt_dir + "/summary.json", "w") as f:
        json.dump(summary_dict, f, indent=4, cls=NumpyEncoder)

    # save numbers
    saving_dict = {"z": z, "z_valid": z_valid, "sample_z": sample_z, "sample_x": sample_x,
                   "sample_z_center": sample_z_center, "sample_x_center": sample_x_center}
    saving_dict = {**dict_of_x_predict_k, **saving_dict}
    if isinstance(tran, LSTMBasedTransformation):
        saving_dict["samples_on_fixed_zs"] = samples_on_fixed_zs

    joblib.dump(saving_dict, rslt_dir + "/numbers")

    # save figures
    plot_z(z, K, title="most likely z for the ground truth")
    plt.savefig(rslt_dir + "/z.jpg")
    plt.close()

    if len(valid_data) >0:
        plot_z(z_valid, K, title="most likely z for valid data")
        plt.savefig(rslt_dir + "/z_valid.jpg")
        plt.close()

    if not os.path.exists(rslt_dir + "/samples"):
        os.makedirs(rslt_dir + "/samples")
        print("Making samples directory...")

    plot_z(sample_z, K, title="sample")
    plt.savefig(rslt_dir + "/samples/sample_z_{}.jpg".format(sample_T))
    plt.close()

    plot_z(sample_z_center, K, title="sample (starting from center)")
    plt.savefig(rslt_dir + "/samples/sample_z_center_{}.jpg".format(sample_T))
    plt.close()

    plt.figure(figsize=(4, 4))
    plot_mouse(data, title="ground truth (training)", xlim=[ARENA_XMIN - 20, ARENA_YMAX + 20],
               ylim=[ARENA_YMIN - 20, ARENA_YMAX + 20])
    plt.legend()
    plt.savefig(rslt_dir + "/samples/ground_truth.jpg")
    plt.close()

    if len(valid_data) > 0:
        plt.figure(figsize=(4, 4))
        plot_mouse(valid_data, title="ground truth (valid)", xlim=[ARENA_XMIN - 20, ARENA_YMAX + 20],
                   ylim=[ARENA_YMIN - 20, ARENA_YMAX + 20])
        plt.legend()
        plt.savefig(rslt_dir + "/samples/ground_truth_valid.jpg")
        plt.close()

    plt.figure(figsize=(4, 4))
    plot_mouse(sample_x, title="sample", xlim=[ARENA_XMIN - 20, ARENA_YMAX + 20],
               ylim=[ARENA_YMIN - 20, ARENA_YMAX + 20])
    plt.legend()
    plt.savefig(rslt_dir + "/samples/sample_x_{}.jpg".format(sample_T))
    plt.close()

    plt.figure(figsize=(4, 4))
    plot_mouse(sample_x_center, title="sample (starting from center)", xlim=[ARENA_XMIN - 20, ARENA_YMAX + 20],
               ylim=[ARENA_YMIN - 20, ARENA_YMAX + 20])
    plt.legend()
    plt.savefig(rslt_dir + "/samples/sample_x_center_{}.jpg".format(sample_T))
    plt.close()

    plot_realdata_quiver(data, z, K, x_grids, y_grids, title="ground truth (training)")
    plt.savefig(rslt_dir + "/samples/quiver_ground_truth.jpg", dpi=200)
    plt.close()

    if len(valid_data) > 0:
        plot_realdata_quiver(valid_data, z_valid, K, x_grids, y_grids, title="ground truth (valid)")
        plt.savefig(rslt_dir + "/samples/quiver_ground_truth_valid.jpg", dpi=200)
        plt.close()

    plot_realdata_quiver(sample_x, sample_z, K, x_grids, y_grids, title="sample")
    plt.savefig(rslt_dir + "/samples/quiver_sample_x_{}.jpg".format(sample_T), dpi=200)
    plt.close()

    plot_realdata_quiver(sample_x_center, sample_z_center, K, x_grids, y_grids, title="sample (starting from center)")
    plt.savefig(rslt_dir + "/samples/quiver_sample_x_center_{}.jpg".format(sample_T), dpi=200)
    plt.close()

    if isinstance(tran, (LinearGridTransformation, GPGridTransformation, WeightedGridTransformation)):
        if not os.path.exists(rslt_dir + "/dynamics"):
            os.makedirs(rslt_dir + "/dynamics")
            print("Making dynamics directory...")

        plot_quiver(XY_grids[:, 0:2], dXY[..., 0:2], 'virgin', K=K, scale=quiver_scale, alpha=0.9,
                    title="quiver (virgin)", x_grids=x_grids, y_grids=y_grids, grid_alpha=0.2)
        plt.savefig(rslt_dir + "/dynamics/quiver_a.jpg", dpi=200)
        plt.close()

        plot_quiver(XY_grids[:, 2:4], dXY[..., 2:4], 'mother', K=K, scale=quiver_scale, alpha=0.9,
                    title="quiver (mother)", x_grids=x_grids, y_grids=y_grids, grid_alpha=0.2)
        plt.savefig(rslt_dir + "/dynamics/quiver_b.jpg", dpi=200)
        plt.close()
    elif isinstance(tran, LSTMBasedTransformation):
        if not os.path.exists(rslt_dir + "/dynamics"):
            os.makedirs(rslt_dir + "/dynamics")
            print("Making dynamics directory...")

        for k in range(K):
            plot_realdata_quiver(samples_on_fixed_zs[k], np.ones(dynamics_T, dtype=np.int)*k, K, x_grids=x_grids, y_grids=y_grids,
                                 title="sample conditioned on k={}".format(k))
            plt.savefig(rslt_dir + "/dynamics/samples_on_k{}.jpg".format(k), dpi=200)
            plt.close()

    if not os.path.exists(rslt_dir + "/distributions"):
        os.makedirs(rslt_dir + "/distributions")
        print("Making distributions directory...")

    # sanity checks
    plot_data_condition_on_all_zs(data, z, K, size=2, alpha=0.3)
    plt.savefig(rslt_dir + "/distributions/spatial_occup_groundtruth.jpg", dpi=100)
    plot_data_condition_on_all_zs(sample_x, sample_z, K, size=2, alpha=0.3)
    plt.savefig(rslt_dir + "/distributions/spatial_occup_sample_x.jpg", dpi=100)
    plot_data_condition_on_all_zs(sample_x_center, sample_z_center, K, size=2, alpha=0.3)
    plt.savefig(rslt_dir + "/distributions/spatial_occup_sample_x_center.jpg", dpi=100)

    plot_2d_time_plot_condition_on_all_zs(data, z, K, title='ground truth')
    plt.savefig(rslt_dir + "/distributions/4traces_groundtruth.jpg", dpi=100)
    plot_2d_time_plot_condition_on_all_zs(sample_x, sample_z, K, title='sample_x')
    plt.savefig(rslt_dir + "/distributions/4traces_sample_x.jpg", dpi=100)
    plot_2d_time_plot_condition_on_all_zs(sample_x_center, sample_z_center, K, title='sample_x_center')
    plt.savefig(rslt_dir + "/distributions/4traces_sample_x_center.jpg", dpi=100)

    data_angles_a, data_angles_b = get_all_angles(data, x_grids, y_grids, device=device)
    sample_angles_a, sample_angles_b = get_all_angles(sample_x, x_grids, y_grids, device=device)
    sample_x_center_angles_a, sample_x_center_angles_b = get_all_angles(sample_x_center, x_grids, y_grids,
                                                                        device=device)

    plot_list_of_angles([data_angles_a, sample_angles_a, sample_x_center_angles_a],
                        ['data', 'sample', 'sample_c'], "direction distribution (virgin)", n_x, n_y)
    plt.savefig(rslt_dir + "/distributions/angles_a.jpg")
    plt.close()
    plot_list_of_angles([data_angles_b, sample_angles_b, sample_x_center_angles_b],
                        ['data', 'sample', 'sample_c'], "direction distribution (mother)", n_x, n_y)
    plt.savefig(rslt_dir + "/distributions/angles_b.jpg")
    plt.close()

    data_speed_a, data_speed_b = get_speed(data, x_grids, y_grids, device=device)
    sample_speed_a, sample_speed_b = get_speed(sample_x, x_grids, y_grids, device=device)
    sample_x_center_speed_a, sample_x_center_speed_b = get_speed(sample_x_center, x_grids, y_grids, device=device)

    plot_list_of_speed([data_speed_a, sample_speed_a, sample_x_center_speed_a],
                       ['data', 'sample', 'sample_c'], "speed distribution (virgin)", n_x, n_y)
    plt.savefig(rslt_dir + "/distributions/speed_a.jpg")
    plt.close()
    plot_list_of_speed([data_speed_b, sample_speed_b, sample_x_center_speed_b],
                       ['data', 'sample', 'sample_c'], "speed distribution (mother)", n_x, n_y)
    plt.savefig(rslt_dir + "/distributions/speed_b.jpg")
    plt.close()

    try:
        if 100 < data.shape[0] <= 36000:
            plot_space_dist(data, x_grids, y_grids)
        elif data.shape[0] > 36000:
            plot_space_dist(data[:36000], x_grids, y_grids)
        plt.savefig(rslt_dir + "/distributions/space_data.jpg")
        plt.close()

        if 100 < sample_x.shape[0] <= 36000:
            plot_space_dist(sample_x, x_grids, y_grids)
        elif sample_x.shape[0] > 36000:
            plot_space_dist(sample_x[:36000], x_grids, y_grids)
        plt.savefig(rslt_dir + "/distributions/space_sample_x.jpg")
        plt.close()

        if 100 < sample_x_center.shape[0] <= 36000:
            plot_space_dist(sample_x_center, x_grids, y_grids)
        elif sample_x_center.shape[0] > 36000:
            plot_space_dist(sample_x_center[:36000], x_grids, y_grids)
        plt.savefig(rslt_dir + "/distributions/space_sample_x_center.jpg")
        plt.close()
    except:
        print("plot_space_dist unsuccessful")

