"""SO(3)/SE(3) helpers: rotation-vector <-> matrix maps, 4x4 <-> 6-vector packing.

The rotation vector is axis * angle, in radians; linear terms are millimetres.
"""

import numpy as np


def rotation_log_map(R_mat):
    """3x3 rotation matrix -> rotation vector (axis * angle, radians)."""
    cosang = float(np.clip((np.trace(R_mat) - 1.0) / 2.0, -1.0, 1.0))
    theta = np.arccos(cosang)
    if theta < 1e-8:
        skew = (R_mat - R_mat.T) / 2.0
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    if cosang < -1.0 + 1e-12:
        # Rotation by ~pi: sin(theta) -> 0 takes the scale below with it, so
        # recover the axis as the +1 eigenvector of the symmetric part instead.
        w, V = np.linalg.eigh((R_mat + R_mat.T) / 2.0)
        axis = V[:, int(np.argmax(w))]
        return np.pi * (-axis if axis[0] < 0.0 else axis)
    scale = theta / (2.0 * np.sin(theta))
    return np.array([R_mat[2, 1] - R_mat[1, 2],
                     R_mat[0, 2] - R_mat[2, 0],
                     R_mat[1, 0] - R_mat[0, 1]]) * scale


def rotation_vector(R_mat):
    """3x3 rotation matrix -> rotation vector in DEGREES, as the controller shows it."""
    return np.degrees(rotation_log_map(R_mat))


def rotation_exp_map(rot_vec):
    """Rotation vector (axis * angle, radians) -> 3x3 rotation matrix (Rodrigues)."""
    angle = np.linalg.norm(rot_vec)
    if angle < 1e-10:
        return np.eye(3)
    axis = rot_vec / angle
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    # Grouped as (1 - cos) * K @ K, not (1 - cos) * (K @ K): the two round
    # differently, and the reference run uses this grouping.
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def t6c_from_vec(vec):
    """Hand-eye 6-vector [x, y, z mm, rx, ry, rz rad] -> T_6_C (4x4, mm)."""
    T = np.eye(4)
    T[:3, 3] = vec[:3]
    T[:3, :3] = rotation_exp_map(vec[3:])
    return T


def vec_from_transform(T):
    """T (4x4, mm) -> 6-vector [x, y, z mm, rx, ry, rz rad]."""
    return np.concatenate([T[:3, 3], rotation_log_map(T[:3, :3])])
