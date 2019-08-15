from ssm_ptc.models.hmm import HMM
from ssm_ptc.observations.ar_gaussian_observation import ARGaussianObservation
from ssm_ptc.transformations.linear import LinearTransformation

def load_data():
    pass


K = 5
D = 2
T = 10

num_iters = 50


tran = LinearTransformation(K=K, d_in=D, D=D)
observation = ARGaussianObservation(K=K, D=D, M=0, transformation=tran)

model = HMM(K=K, D=D, M=0, observation=observation)

data = HMM.sample(T)
