import torch
import numpy as np

from ssm_ptc.transitions.sticky_transition import dirichlet_logpdf

from ssm_ptc.models.hmm import HMM

from torch.distributions.dirichlet import Dirichlet


def test_dirichlet_logpdf():
    alpha = torch.tensor([0.5, 0.6, 1.2])
    ps = torch.tensor([0.2, 0.3, 0.5])

    log_pdf = dirichlet_logpdf(ps, alpha)

    # pytorch implementation
    dist = Dirichlet(concentration=alpha)
    log_prob = dist.log_prob(ps)

    print(log_pdf)
    print(log_prob)

    assert log_pdf == log_prob


def test_stick_transition():
    K = 3
    D = 2
    M = 0

    model = HMM(K=K, D=D, M=M, transition='sticky', observation='gaussian', transition_kwargs=dict(alpha=1, kappa=100))
    print("alpha = {}, kappa = {}".format(model.transition.alpha, model.transition.kappa))

    data = torch.tensor([[2,3], [4,5], [6, 7]], dtype=torch.float64)

    log_prob = model.log_likelihood(data)
    print("log_prob", log_prob)

    samples_z, samples_x = model.sample(10)

    samples_log_prob = model.log_probability(samples_x)
    print("sample log_prob", samples_log_prob)

    print("model transition\n", model.transition.stationary_transition_matrix)

    model1 = HMM(K=K, D=D, M=M, transition='sticky', observation='gaussian', transition_kwargs=dict(alpha=1, kappa=0.1))

    print("Before fit")
    print("model1 transition\n", model1.transition.stationary_transition_matrix)

    losses, opt = model1.fit(samples_x)
    print("After fit")
    print("model1 transition\n", model1.transition.stationary_transition_matrix)

    model2 = HMM(K=K, D=D, M=M, transition='sticky', observation='gaussian', transition_kwargs=dict(alpha=1, kappa=20))

    print("Before fit")
    print("model2 transition\n", model2.transition.stationary_transition_matrix)

    losses, opt = model2.fit(samples_x)
    print("After fit")
    print("model2 transition\n", model2.transition.stationary_transition_matrix)

    model3 = HMM(K=K, D=D, M=M, transition='sticky', observation='gaussian', transition_kwargs=dict(alpha=1, kappa=100))

    print("Before fit")
    print("model3 transition\n", model3.transition.stationary_transition_matrix)

    losses, opt = model3.fit(samples_x)
    print("After fit")
    print("model3 transition\n", model3.transition.stationary_transition_matrix)


# test_dirichlet_logpdf()
test_stick_transition()

