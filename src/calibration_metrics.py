"""Scoring: how consistent a calibration is, and how two calibrations differ."""

import numpy as np

from calibration_model import predict_all_T0W, predict_all_T0W_from_flange
from fk_lite6 import fk_from_zeta_batch
from so3 import rotation_log_map, t6c_from_vec


def stats_from_T0W(T0W):
    """Consistency stats for a stack of T_0_W: (l,4,4) -> pos_axis_std_mm, s_max_mm,
    s_max_axis, trans_max_mm, ori_mean_deg, ori_max_deg."""

    pos = T0W[:, :3, 3]
    R0 = T0W[0, :3, :3]
    rel_ang = np.array([np.linalg.norm(rotation_log_map(R0.T @ T0W[i, :3, :3]))
                        for i in range(T0W.shape[0])])
    std = pos.std(axis=0, ddof=0)

    return {
        'pos_axis_std_mm': std,
        's_max_mm': float(std.max()),
        's_max_axis': 'XYZ'[int(np.argmax(std))],
        'trans_max_mm': float(np.max(np.linalg.norm(pos - pos.mean(axis=0), axis=1))),
        'ori_mean_deg': float(np.rad2deg(rel_ang.mean())),
        'ori_max_deg': float(np.rad2deg(rel_ang.max())),
    }


def compute_T0W_stats(Q, TCW_list, xi, n=6):
    """stats_from_T0W over T_0_W predicted from DH parameters plus hand-eye."""
    return stats_from_T0W(predict_all_T0W(Q, TCW_list, xi, n))


def compute_T0W_stats_from_flange(T_0_6_all, TCW_list, t6c_vec):
    """stats_from_T0W with T_0_6 supplied, so no DH error is charged."""
    return stats_from_T0W(predict_all_T0W_from_flange(T_0_6_all, TCW_list, t6c_vec))


def artifact_displacement(T_0_6_before, TCW_before, T_0_6_after, TCW_after, t6c_vec):
    """Mean W position in two captures and the distance between them, chained
    through the T_0_6 passed in."""

    before = predict_all_T0W_from_flange(T_0_6_before, TCW_before, t6c_vec)
    after = predict_all_T0W_from_flange(T_0_6_after, TCW_after, t6c_vec)
    b, a = before[:, :3, 3].mean(axis=0), after[:, :3, 3].mean(axis=0)
    return {'before_mm': b, 'after_mm': a, 'moved_mm': float(np.linalg.norm(a - b))}


def fk_vs_reference_rmse(Q, T_ref_list, zeta, n=6):
    """Our FK vs a reference flange stack: (l,n) rad + (l,4,4) + zeta -> rmse_mm,
    pos_std_mm, ori_mean_deg, ori_max_deg."""

    T_ref_list = np.asarray(T_ref_list, dtype=float)
    T_0_6 = fk_from_zeta_batch(Q, zeta, n)
    dp = T_0_6[:, :3, 3] - T_ref_list[:, :3, 3]
    dq = np.array([np.linalg.norm(rotation_log_map(
        T_0_6[i, :3, :3].T @ T_ref_list[i, :3, :3])) for i in range(Q.shape[0])])

    return {
        'rmse_mm': float(np.sqrt((dp ** 2).sum(axis=1).mean())),
        'pos_std_mm': float(dp.std(axis=0, ddof=0).max()),
        'ori_mean_deg': float(np.rad2deg(dq.mean())),
        'ori_max_deg': float(np.rad2deg(dq.max())),
    }


def t6c_difference(t6c_a, t6c_b):
    """Translation (mm) and geodesic rotation (deg) between two T_6_C vectors."""
    dt = np.linalg.norm(np.asarray(t6c_a, dtype=float)[:3]
                        - np.asarray(t6c_b, dtype=float)[:3])
    R_a = t6c_from_vec(np.asarray(t6c_a, dtype=float))[:3, :3]
    R_b = t6c_from_vec(np.asarray(t6c_b, dtype=float))[:3, :3]
    dq = np.linalg.norm(rotation_log_map(R_a.T @ R_b))
    return float(dt), float(np.rad2deg(dq))
