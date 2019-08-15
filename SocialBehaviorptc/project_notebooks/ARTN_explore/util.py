import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt


def plot_2_mice(data):
    plt.plot(data[:,0], data[:,1], label='virgin')
    plt.plot(data[:,2], data[:,3], label='mother')
    plt.legend();


def plot_4_traces(data):
    plt.plot(data[:, 0], label='x1')
    plt.plot(data[:, 1], label='y1')
    plt.plot(data[:, 2], label='x2')
    plt.plot(data[:, 3], label='y2')
    plt.legend();


def sigmoid(x):
    return 1. / (1+np.exp(-x))


def inv_sigmoid(x):
    return np.log(x) - np.log(1-x)


def transform(x, bounds, scale=1, offset=0):
    """
    [0, max] -> (-infty, infty)

    scale: (4,) or scalar
    offset: (4,) or scalar
    """
    h = (x - bounds[...,0]) / (bounds[...,1] - bounds[...,0]) # [min, max] --> [0,1]
    y = inv_sigmoid(h) / scale + offset # [0, 1] --> (-infty, infty)
    return y


def inv_transform(y, bounds, scale=1, offset=0):
    """
    (-infty, infty) -> [0, max]
    """
    h = sigmoid(scale * (y - offset)) # (-infty, infty) --> [0,1]
    x = (bounds[...,1] - bounds[...,0]) * h + bounds[...,0]  #  [0,1] --> [min, max]
    return x


def k_step_prediction(model, model_z, data, k=0):
    """
    Conditioned on the most likely hidden states, make the k-step prediction.
    """
    x_predict_arr = []
    input = np.zeros((data.shape[0],))
    if k == 0:
        for t in range(data.shape[0]):
            x_predict = model.observations.sample_x(model_z[t], data[:t], input)
            x_predict_arr.append(x_predict)
    else:
        assert k>0
        # neglects t = 0 since there is no history
        for t in range(1, data.shape[0]-k):
            zx_predict = model.sample(k, prefix=(model_z[t-1:t], data[t-1:t]))
            assert zx_predict[1].shape == (k, 4)
            x_predict = zx_predict[1][k-1]
            x_predict_arr.append(x_predict)
    x_predict_arr = np.array(x_predict_arr)
    return x_predict_arr
