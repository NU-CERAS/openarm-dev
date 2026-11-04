import numpy as np


def score_observation_scales(include_proprio=False):
    scales = np.ones(269, dtype=np.float32)
    for frame in range(4):
        scales[frame*42:frame*42+14] = .03
        scales[frame*42+14:frame*42+28] = .5
        scales[frame*42+28:frame*42+42] = .03
    scales[168:189] = .01
    return np.r_[scales, np.full(14, .1)].astype(np.float32) if include_proprio else scales
