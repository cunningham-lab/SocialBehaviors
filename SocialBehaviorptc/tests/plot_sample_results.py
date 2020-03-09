import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import joblib
import numpy as np
import time

from project_ssms.constants import *
from project_ssms.plot_utils import animate
from project_ssms.utils import downsample

K = 6

data_path = "/Users/leah/Columbia/courses/19summer/SocialBehavior/SocialBehaviorptc/data/trajs_all"
#numbers_path = "/Users/leah/Columbia/courses/19summer/SocialBehavior/rslts/gp/1126_transition/v05_mother_downsamplen2_grid_K6_6by6_D191125_204944/checkpoint_1/numbers"
numbers_path = "/Users/leah/Columbia/courses/19summer/SocialBehavior/rslts/gp/1126_transition/v05_mother_downsamplen2_stationary_K6_6by6_D191125_204944/checkpoint_0/numbers"
numbers = joblib.load(numbers_path)

start = 0
end = 5

"""
# real data
traj = joblib.load(data_path)
traj = traj[int(36000 * start): int(36000 * end)]
traj = traj[:,2:4]
downsample_n = 2
traj = downsample(traj, downsample_n)

z = numbers['z']
t = len(z)
traj = traj[:t]

downsampled_factor = 10
downsampled_range = np.arange(0, t, downsampled_factor)
downsampled_traj = traj[downsampled_range]
downsampled_z = z[downsampled_range]

print("shape", downsampled_traj.shape)
downsample_t, _ = downsampled_traj.shape

start = time.time()
animate(downsampled_traj, downsampled_z, 6,
        xlim=[ARENA_XMIN-10, ARENA_XMAX+10], ylim=[ARENA_YMIN-10, ARENA_YMAX+10])

end = time.time()
print("elapsed time {} for {} datapoints".format(end-start, downsample_t))
"""
# sample data
sample_x = numbers['sample_x']
sample_z = numbers['sample_z']

print("shape", sample_x.shape, sample_z.shape)

t, _ = sample_x.shape

downsampled_factor = 10  
downsampled_range = np.arange(0, t, downsampled_factor)
downsampled_sample_x = sample_x[downsampled_range]
downsampled_sample_z = sample_z[downsampled_range]

print("shape", downsampled_sample_x.shape, downsampled_sample_z.shape)

downsample_t, _ = downsampled_sample_x.shape


start = time.time()
animate(downsampled_sample_x, downsampled_sample_z, 6,
        xlim=[ARENA_XMIN-10, ARENA_XMAX+10], ylim=[ARENA_YMIN-10, ARENA_YMAX+10])

end = time.time()
print("elapsed time {} for {} datapoints".format(end-start, downsample_t))

# elapsed time 10.744312286376953 for 87 datapoints
