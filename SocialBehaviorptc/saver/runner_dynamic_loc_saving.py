import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib

from project_ssms.plot_utils import plot_z, plot_2_mice
from project_ssms.grid_utils import plot_quiver, plot_realdata_quiver, \
    get_all_angles, get_speed, plot_list_of_angles, plot_list_of_speed, plot_space_dist
from project_ssms.constants import *

from ssm_ptc.utils import k_step_prediction

from saver.rslts_saving import NumpyEncoder


def rslt_saving(rslt_dir, model, data, sample_T, train_model, losses, quiver_scale, x_grids, y_grids):
    tran = model.observation.transformation
    n_x = len(x_grids) - 1
    n_y = len(y_grids) - 1

    K = model.K



    #################### inference ###########################

    print("\ninferring most likely states...")
    z = model.most_likely_states(data)

    print("0 step prediction")
    if data.shape[0] <= 5000:
        data_to_predict = data
    else:
        data_to_predict = data[-5000:]
    x_predict = k_step_prediction(model, z, data_to_predict)
    x_predict_err = np.mean(np.abs(x_predict - data_to_predict.numpy()), axis=0)

    print("5 step prediction")
    x_predict_5 = k_step_prediction(model, z, data_to_predict, k=5)
    x_predict_5_err = np.mean(np.abs(x_predict_5 -data_to_predict[5:].numpy()), axis=0)


    ################### samples #########################

    sample_z, sample_x = model.sample(sample_T)

    center_z = torch.tensor([0], dtype=torch.int)
    center_x = torch.tensor([[150, 190, 200, 200]], dtype=torch.float64)
    sample_z_center, sample_x_center = model.sample(sample_T, prefix=(center_z, center_x))

    ################## dynamics #####################

    # quiver
    XX, YY = np.meshgrid(np.linspace(20, 310, 30),
                         np.linspace(0, 380, 30))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY)))  # shape (900,2) grid values
    XY_grids = np.concatenate((XY, XY), axis=1)

    XY_next = tran.transform(torch.tensor(XY_grids, dtype=torch.float64))
    dXY = XY_next.detach().numpy() - XY_grids[:, None]

    #################### saving ##############################

    print("begin saving...")

    # save summary
    avg_transform_speed = np.average(np.abs(dXY), axis=0)
    avg_sample_speed = np.average(np.abs(np.diff(sample_x, axis=0)), axis=0)
    avg_sample_center_speed = np.average(np.abs(np.diff(sample_x_center, axis=0)), axis=0)
    avg_data_speed = np.average(np.abs(np.diff(data.numpy(), axis=0)), axis=0)

    transition_matrix = model.transition.stationary_transition_matrix
    if transition_matrix.requires_grad:
        transition_matrix = transition_matrix.detach().numpy()
    else:
        transition_matrix = transition_matrix.numpy()


    summary_dict = {"init_dist": model.init_dist.detach().numpy(),
                    "transition_matrix": transition_matrix,
                    "x_predict_err": x_predict_err, "x_predict_5_err": x_predict_5_err,
                    "variance": torch.exp(model.observation.log_sigmas).detach().numpy(),
                    "log_likes": model.log_likelihood(data).detach().numpy(),
                    "avg_transform_speed": avg_transform_speed, "avg_data_speed": avg_data_speed,
                    "avg_sample_speed": avg_sample_speed, "avg_sample_center_speed": avg_sample_center_speed
                    }
    with open(rslt_dir + "/summary.json", "w") as f:
        json.dump(summary_dict, f, indent=4, cls=NumpyEncoder)

    # save numbers
    saving_dict = {"z": z, "x_predict": x_predict, "x_predict_5": x_predict_5,
                   "sample_z": sample_z, "sample_x": sample_x,
                   "sample_z_center": sample_z_center, "sample_x_center": sample_x_center}
    
    if train_model:
        saving_dict['losses'] = losses
        plt.figure()
        plt.plot(losses)
        plt.savefig(rslt_dir + "/losses.jpg")
        plt.close()
    joblib.dump(saving_dict, rslt_dir + "/numbers")

    # save figures
    plot_z(z, K, title="most likely z for the ground truth")
    plt.savefig(rslt_dir + "/z.jpg")
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
    plot_2_mice(data, title="ground truth", xlim=[ARENA_XMIN - 20, ARENA_XMAX + 20],
                ylim=[ARENA_YMIN - 20, ARENA_YMAX + 20])
    plt.legend()
    plt.savefig(rslt_dir + "/samples/ground_truth.jpg")
    plt.close()

    plt.figure(figsize=(4, 4))
    plot_2_mice(sample_x, title="sample", xlim=[ARENA_XMIN - 20, ARENA_XMAX + 20],
                ylim=[ARENA_YMIN - 20, ARENA_YMAX + 20])
    plt.legend()
    plt.savefig(rslt_dir + "/samples/sample_x_{}.jpg".format(sample_T))
    plt.close()

    plt.figure(figsize=(4, 4))
    plot_2_mice(sample_x_center, title="sample (starting from center)",
                xlim=[ARENA_XMIN - 20, ARENA_XMAX + 20], ylim=[ARENA_YMIN - 20, ARENA_YMAX + 20])
    plt.legend()
    plt.savefig(rslt_dir + "/samples/sample_x_center_{}.jpg".format(sample_T))
    plt.close()

    plot_realdata_quiver(data, z, K, x_grids, y_grids, scale=1, title="ground truth")
    plt.savefig(rslt_dir + "/samples/quiver_ground_truth.jpg", dpi=200)

    plot_realdata_quiver(sample_x, sample_z, K, x_grids, y_grids, scale=1, title="sample")
    plt.savefig(rslt_dir + "/samples/quiver_sample_x_{}.jpg".format(sample_T), dpi=200)
    plt.close()

    plot_realdata_quiver(sample_x_center, sample_z_center, K, x_grids, y_grids, scale=1, title="sample (starting from center)")
    plt.savefig(rslt_dir + "/samples/quiver_sample_x_center_{}.jpg".format(sample_T), dpi=200)
    plt.close()

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

    if not os.path.exists(rslt_dir + "/distributions"):
        os.makedirs(rslt_dir + "/distributions")
        print("Making distributions directory...")

    data_angles_a, data_angles_b = get_all_angles(data, x_grids, y_grids)
    sample_angles_a, sample_angles_b = get_all_angles(sample_x, x_grids, y_grids)
    sample_x_center_angles_a, sample_x_center_angles_b = get_all_angles(sample_x_center, x_grids, y_grids)

    plot_list_of_angles([data_angles_a, sample_angles_a, sample_x_center_angles_a],
                        ['data', 'sample', 'sample_c'], "direction distribution (virgin)", n_x, n_y)
    plt.savefig(rslt_dir + "/distributions/angles_a.jpg")
    plt.close()
    plot_list_of_angles([data_angles_b, sample_angles_b, sample_x_center_angles_b],
                        ['data', 'sample', 'sample_c'], "direction distribution (mother)", n_x, n_y)
    plt.savefig(rslt_dir + "/distributions/angles_b.jpg")
    plt.close()

    data_speed_a, data_speed_b = get_speed(data, x_grids, y_grids)
    sample_speed_a, sample_speed_b = get_speed(sample_x, x_grids, y_grids)
    sample_x_center_speed_a, sample_x_center_speed_b = get_speed(sample_x_center, x_grids, y_grids)

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

