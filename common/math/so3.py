"""
Rotation related functions for numpy arrays
"""

import numpy as np
from scipy.spatial.transform import Rotation


def dcm2euler(mats: np.ndarray, seq: str = 'zyx', degrees: bool = True):
    """Converts rotation matrix to euler angles

    Args:
        mats: (B, 3, 3) containing the B rotation matricecs
        seq: Sequence of euler rotations (default: 'zyx')
        degrees (bool): If true (default), will return in degrees instead of radians

    Returns:

    """

    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=degrees))
    return np.stack(eulers)


def transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SO3 transform

    Args:
        g: SO3 transformation matrix of size (3, 3)
        pts: Points to be transformed (N, 3)

    Returns:
        transformed points of size (N, 3)

    """
    rot = g[:3, :3]  # (3, 3)
    transformed = pts @ rot.transpose()
    return transformed
