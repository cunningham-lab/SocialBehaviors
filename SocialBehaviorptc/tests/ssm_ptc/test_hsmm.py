import torch
import numpy as np

from ssm_ptc.models.hsmm import HSMM


def test_marginalization():
    from project_ssms.gp_observation_single import GPObservationSingle
    device = torch.device("cpu")

    torch.random.manual_seed(0)
    np.random.seed(0)

    data = torch.tensor([[0, 0], [5, 5], [10, 10]], dtype=torch.float64)
    #zs = torch.tensor([0, 1, 1], dtype=torch.int)
    #Ls = torch.tensor([1, 2, 1], dtype=torch.int)

    T = 3
    D = 2
    K = 2

    n_x = 3
    n_y = 3

    ARENA_XMIN, ARENA_XMAX = -20, 20
    ARENA_YMIN, ARENA_YMAX = -20, 20
    mus_init = data[0] * torch.ones(K, D, dtype=torch.float64, device=device)
    x_grid_gap = (ARENA_XMAX - ARENA_XMIN) / n_x
    x_grids = np.array([ARENA_XMIN + i * x_grid_gap for i in range(n_x + 1)])
    y_grid_gap = (ARENA_YMAX - ARENA_YMIN) / n_y
    y_grids = np.array([ARENA_YMIN + i * y_grid_gap for i in range(n_y + 1)])
    bounds = np.array([[ARENA_XMIN, ARENA_XMAX], [ARENA_YMIN, ARENA_YMAX]])
    train_rs = False
    obs = GPObservationSingle(K=K, D=D, mus_init=mus_init, x_grids=x_grids, y_grids=y_grids, bounds=bounds,
                              rs=None, train_rs=train_rs, device=device)

    model = HSMM(K=K, D=D, L=2, observation=obs)

    log_mariginal_likelihood = model.log_likelihood(data) # -41.7526
    print("log marginal likelihood", log_mariginal_likelihood)

    joint_likes = []
    Ls_list = [torch.tensor([1, 2, 1]), torch.tensor([1, 1, 1]), torch.tensor([2, 1, 1])]
    zs_list = [torch.tensor([0, 0, 0]), torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]), torch.tensor([0, 1, 1]),
          torch.tensor([1, 0, 0]), torch.tensor([1, 0, 1]), torch.tensor([1, 1, 0]), torch.tensor([1, 1, 1])]

    for Ls in Ls_list:
        for zs in zs_list:
            log_joint_likelihood = model.log_joint_likelihood(zs=zs, Ls=Ls, data=data)
            joint_likes.append(log_joint_likelihood)
    joint_likes = torch.stack(joint_likes) # (24, )
    print("joing_likes 0", joint_likes[0]) # -2.1665
    print("joint_likes shape", joint_likes.shape)
    log_marginal_likelihood2 = torch.logsumexp(joint_likes, dim=0)  # -0.8034
    print("log joint likelihood", log_marginal_likelihood2)

    _, zs = model.most_likely_states(data)


if __name__ == "__main__":
    test_marginalization()