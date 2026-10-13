"""The three calibration stages: they estimate, they do not print or save.

The model each stage fits is written out immediately below.
"""

#     Stage 1  T_6_C only, against FK of the nominal DH table.
#     Stage F  T_6_C only, against the controller's own T_0_6 for the same joint angles.
#     Stage 2  T_6_C plus 16 free DH parameters, Siciliano's eight locked at nominal,
#              seeded from Stage 1.

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import lab_config as C
from calibration_metrics import compute_T0W_stats, compute_T0W_stats_from_flange
from calibration_model import (compute_residuals_and_jacobian,
                               hand_eye_residuals_and_jacobian,
                               init_T6C_two_pose, init_T6C_two_pose_from_flange)
from fk_lite6 import locked_indices
from lm_optimizer import levenberg_marquardt


TITLES = {
    'stage1': 'STAGE 1 - HAND-EYE AGAINST NOMINAL KINEMATICS',
    'stagef': 'STAGE F - HAND-EYE AGAINST THE CONTROLLER FK',
    'stage2': 'STAGE 2 - PARTIAL-MODEL DH + HAND-EYE (SICILIANO)',
}


@dataclass
class StageResult:
    """What one stage fitted and how well it scores."""
    t6c: np.ndarray
    stats: dict
    iters: int
    converged: bool
    status: str
    zeta: Optional[np.ndarray] = None      # DH used by this model, None if nominal
    free_idx: Optional[np.ndarray] = None  # zeta indices it was allowed to move
    extra: dict = field(default_factory=dict)


def run_stage1(Q, TCW_list, zeta_nominal, w_rows, n=6, t6c_seed=None, verbose=True):
    """Hand-eye only: 6 parameters, T_0_6 from FK of the nominal DH table."""
    zeta0 = np.asarray(zeta_nominal, dtype=float).copy()
    if t6c_seed is None:
        t6c_seed = init_T6C_two_pose(Q, TCW_list, zeta0, n)

    def residuals(t6c):
        """Weighted T_0_W residual and Jacobian, DH columns sliced off."""
        xi = np.concatenate([zeta0, t6c])
        r, J = compute_residuals_and_jacobian(xi, Q, TCW_list, n)
        return r, J[:, 4 * n:]

    fit = levenberg_marquardt(residuals, t6c_seed, w_rows,
                              max_iter=C.STAGE1_MAX_ITER, tol=C.LM_TOL,
                              lam0=C.LM_LAM0, reg_weight=C.STAGE1_REG_WEIGHT,
                              svd_thresh=C.STAGE1_SVD_THRESH, verbose=verbose)
    t6c = fit['xi']
    return StageResult(
        t6c=t6c,
        stats=compute_T0W_stats(Q, TCW_list, np.concatenate([zeta0, t6c]), n),
        iters=fit['iters'], converged=fit['converged'], status=fit['status'])


def run_stage2(Q, TCW_list, zeta_nominal, w_rows, n=6, locked=None,
               t6c_seed=None, verbose=True):
    """Hand-eye plus DH with a lock list: 16 DH + 6 hand-eye, or all 30 if locked=[].

    locked defaults to lab_config.LOCKED_TABLE_NAMES, Siciliano's eight.
    """

    zeta0 = np.asarray(zeta_nominal, dtype=float).copy()
    if t6c_seed is None:
        t6c_seed = init_T6C_two_pose(Q, TCW_list, zeta0, n)

    lock_idx = (locked_indices(n=n) if locked is None
                else np.asarray(locked, dtype=int))
    free_idx = np.setdiff1d(np.arange(4 * n), lock_idx)
    n_free = len(free_idx)

    def residuals(x_free):
        """Residual and Jacobian in the free-DH + hand-eye subspace."""
        delta_dh = x_free[:n_free]
        t6c = x_free[n_free:]
        zeta = zeta0.copy()
        zeta[free_idx] += delta_dh
        xi = np.concatenate([zeta, t6c])
        r, J = compute_residuals_and_jacobian(xi, Q, TCW_list, n)
        return r, np.hstack([J[:, free_idx], J[:, 4 * n:]])

    x0 = np.concatenate([np.zeros(n_free), t6c_seed])
    fit = levenberg_marquardt(residuals, x0, w_rows,
                              max_iter=C.STAGE2_MAX_ITER, tol=C.LM_TOL,
                              lam0=C.LM_LAM0, reg_weight=C.STAGE2_REG_WEIGHT,
                              svd_thresh=C.STAGE2_SVD_THRESH, verbose=verbose)

    zeta = zeta0.copy()
    zeta[free_idx] += fit['xi'][:n_free]
    t6c = fit['xi'][n_free:]

    # The locked entries of the returned table must equal nominal exactly.
    lock_err = (float(np.max(np.abs(zeta[lock_idx] - zeta0[lock_idx])))
                if len(lock_idx) else 0.0)

    return StageResult(
        t6c=t6c,
        stats=compute_T0W_stats(Q, TCW_list, np.concatenate([zeta, t6c]), n),
        iters=fit['iters'], converged=fit['converged'], status=fit['status'],
        zeta=zeta, free_idx=free_idx,
        extra={'n_free': n_free, 'lock_idx': lock_idx, 'lock_err': float(lock_err),
               'lock_held': lock_err == 0.0})


def run_stage_f(T_0_6_controller, Q, TCW_list, zeta_nominal, w_rows, n=6,
                t6c_seed=None, verbose=True):
    """Hand-eye only against the controller's T_0_6; no DH parameter is fitted."""
    if t6c_seed is None:
        t6c_seed = init_T6C_two_pose_from_flange(T_0_6_controller, TCW_list)

    fit = levenberg_marquardt(
        lambda p: hand_eye_residuals_and_jacobian(p, T_0_6_controller, TCW_list),
        t6c_seed, w_rows,
        max_iter=C.STAGE1_MAX_ITER, tol=C.LM_TOL, lam0=C.LM_LAM0,
        reg_weight=C.STAGE1_REG_WEIGHT, svd_thresh=C.STAGE1_SVD_THRESH,
        verbose=verbose)
    t6c = fit['xi']
    return StageResult(
        t6c=t6c,
        stats=compute_T0W_stats(Q, TCW_list,
                                np.concatenate([zeta_nominal, t6c]), n),
        iters=fit['iters'], converged=fit['converged'], status=fit['status'],
        zeta=np.asarray(zeta_nominal, dtype=float).copy(),
        extra={'stats_controller': compute_T0W_stats_from_flange(
            T_0_6_controller, TCW_list, t6c)})
