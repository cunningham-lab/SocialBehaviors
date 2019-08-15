import numpy as np

import joblib

datasets_processed = joblib.load('/Users/leah/Columbia/courses/19summer/SocialBehavior/tracedata/all_data_3_1')  # a list of length 30, each is a social_dataset


datasets_processed[0].filter_speed(8, threshold=10)

datasets_processed[0].filter_speed(3, threshold=10)

session_data = datasets_processed[0].render_trajectories([3, 8])

traj0 = np.concatenate((session_data), axis=1)

#data = torch.tensor(traj0[:,2:4], dtype=torch.float64)