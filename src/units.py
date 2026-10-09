"""The only places degrees and radians cross over.

Disk is millimetres and degrees (Lab 1 convention); every calculation is millimetres and radians.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_from_pose_deg(pose):
    """Base pose [x, y, z mm, roll, pitch, yaw deg] -> T_0_6 (4x4, mm).

    Controller-reported flange pose; x-y-z Euler, degrees in.
    """

    p = np.asarray(pose, dtype=float).reshape(-1)
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = R.from_euler('xyz', p[3:6], degrees=True).as_matrix()
    return T


def transform_from_pose_rad(pose):
    """Pose [x, y, z mm, roll, pitch, yaw rad] -> T (4x4, mm), same x-y-z order."""
    p = np.asarray(pose, dtype=float).reshape(-1)
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = R.from_euler('xyz', p[3:6]).as_matrix()
    return T


def transform_from_rt(rotation, translation):
    """Rotation matrix (3x3) and translation (3,) -> homogeneous transform (4x4)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=float)
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def euler_deg_from_matrix(M):
    """Rotation matrix -> (roll, pitch, yaw) degrees, x-y-z order."""
    return R.from_matrix(M).as_euler('xyz', degrees=True)


def joints_to_radians(joints_deg):
    """Stored joint angles (l, n) degrees -> radians."""
    return np.deg2rad(np.asarray(joints_deg, dtype=float))
