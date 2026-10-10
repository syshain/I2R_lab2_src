"""Forward model, residuals and hand-eye seeding for the chain T_0_W = T_0_6 @ T_6_C @ T_C_W.

Frames: 0 = base, 6 = flange, C = camera, W = artifact; T_0_W must repeat for every pose.
"""

import numpy as np

import lab_config as C
from fk_lite6 import fk_from_zeta
from so3 import rotation_log_map, t6c_from_vec, vec_from_transform


def predict_all_T0W(Q, TCW_list, xi, n=6):
    """T_0_W per pose from xi = [zeta | t6c]. Q (l,n) rad -> (l, 4, 4) mm."""
    zeta = xi[:4 * n]
    T_6_C = t6c_from_vec(xi[4 * n:])
    l = Q.shape[0]
    T0W = np.empty((l, 4, 4))
    for i in range(l):
        T_0_6 = fk_from_zeta(Q[i], zeta, n)
        T0W[i] = T_0_6 @ T_6_C @ TCW_list[i]
    return T0W


def predict_all_T0W_from_flange(T_0_6_all, TCW_list, t6c_vec):
    """Same chain as predict_all_T0W with T_0_6 supplied instead of fitted.

    (l,4,4) flange + (l,4,4) camera->artifact + 6-vector t6c -> (l, 4, 4) mm.
    """

    T_6_C = t6c_from_vec(t6c_vec)
    T_6_W = T_6_C @ np.asarray(TCW_list, dtype=float)
    return np.asarray(T_0_6_all, dtype=float) @ T_6_W


def consistency_residual(T0W):
    """(l, 4, 4) T_0_W -> stacked residual (6l), pose 0 as the reference.

    Block i is the relative transform T_0_W[0]^-1 T_0_W[i]: its translation
    R_0^T (p_i - p_0) in mm and the rotation vector of R_0^T R_i in rad, so
    block 0 is identically zero. Both halves are taken in the reference frame,
    which is what makes the whole residual invariant to a common rigid motion
    of base and artifact (Section 6.4, first group); a bare p_0 - p_i is not
    invariant to that motion's rotation. Section 5.2 prescribes the rotation
    vector of the relative rotation: it vanishes exactly when two orientations
    agree, which the difference of two axis-angle vectors does not.
    """
    l = T0W.shape[0]
    p0 = T0W[0, :3, 3]
    R0 = T0W[0, :3, :3]
    r = np.empty((l, 6))
    for i in range(l):
        r[i, :3] = R0.T @ (T0W[i, :3, 3] - p0)
        r[i, 3:] = rotation_log_map(R0.T @ T0W[i, :3, :3])
    return r.flatten()


def hand_eye_residuals_and_jacobian(t6c_vec, T_0_6_all, TCW_list, eps=C.FD_EPS):
    """Residual + Jacobian for hand-eye-only fitting against given flange poses.

    Returns (r, J) with r length 6*l and J (6*l, 6); J = -dr/dt6c.
    """

    r = consistency_residual(predict_all_T0W_from_flange(T_0_6_all, TCW_list, t6c_vec))

    J = np.zeros((len(TCW_list) * 6, 6))
    for j in range(6):
        xp = t6c_vec.copy()
        xp[j] += eps
        rp = consistency_residual(predict_all_T0W_from_flange(T_0_6_all, TCW_list, xp))
        J[:, j] = -(rp - r) / eps
    return r, J


def compute_residuals_and_jacobian(xi, Q, TCW_list, n=6, eps=C.FD_EPS):
    """Residual + finite-difference Jacobian of the T_0_W consistency cost.

    (r, J): r length 6*l with block 0 identically zero, J = -dr/dxi.
    """

    r = consistency_residual(predict_all_T0W(Q, TCW_list, xi, n))

    J = np.zeros((Q.shape[0] * 6, len(xi)))
    for j in range(len(xi)):
        xp = xi.copy()
        xp[j] += eps
        rp = consistency_residual(predict_all_T0W(Q, TCW_list, xp, n))
        J[:, j] = -(rp - r) / eps
    return r, J


def _orthogonalise_rotation(M):
    """Closest proper rotation to a 3x3 matrix, by SVD."""
    U, _, Vt = np.linalg.svd(M)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        Vt[-1, :] *= -1
        R_proj = U @ Vt
    return R_proj


def init_T6C_two_pose_from_flange(T_0_6_all, TCW_list):
    """Seed T_6_C from the most widely separated of the first ten views.

    (l,4,4) flange + (l,4,4) camera->artifact -> 6-vector; zeros on failure.
    """

    try:
        T_0_6_all = np.asarray(T_0_6_all, dtype=float)
        l = len(TCW_list)

        best_gap = 0
        i_ref, j_ref = 0, 1
        for i in range(min(l, 10)):
            for j in range(i + 1, min(l, 10)):
                gap = np.linalg.norm(TCW_list[i][:3, 3] - TCW_list[j][:3, 3])
                if gap > best_gap:
                    best_gap = gap
                    i_ref, j_ref = i, j

        # T_6_C ~ identity gives the artifact pose from view i_ref; that plus
        # each flange pose implies a T_6_C at both extreme views.
        T_0_W_guess = T_0_6_all[i_ref] @ TCW_list[i_ref]
        T_6_C_a = np.linalg.inv(T_0_6_all[i_ref]) @ T_0_W_guess @ np.linalg.inv(TCW_list[i_ref])
        T_6_C_b = np.linalg.inv(T_0_6_all[j_ref]) @ T_0_W_guess @ np.linalg.inv(TCW_list[j_ref])
        T_6_C = (T_6_C_a + T_6_C_b) / 2.0

        T_6_C[:3, :3] = _orthogonalise_rotation(T_6_C[:3, :3])
        return vec_from_transform(T_6_C)
    except Exception:
        return np.zeros(6)


def init_T6C_two_pose(Q, TCW_list, zeta_nom, n=6):
    """init_T6C_two_pose_from_flange with T_0_6 from nominal-DH forward kinematics."""
    T_0_6_all = np.array([fk_from_zeta(Q[i], zeta_nom, n) for i in range(Q.shape[0])])
    return init_T6C_two_pose_from_flange(T_0_6_all, TCW_list)
