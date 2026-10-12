"""Task 1: numerical inverse kinematics for the Lite 6, against fk_lite6.

Units are the ones the controller displays: angles in degrees, positions in
millimetres (handout section 5.2).

    python3 ik_lite6.py                 the residual self-check of Task 1(c)
    python3 ik_lite6.py X Y Z R P Y     solve one pose (mm, degrees) from rest
    python3 ik_lite6.py X Y Z R P Y q1 ... q6
                        solve from a starting guess, inside the joint box

The solver is a damped Levenberg-Marquardt loop through an SVD. The part you
write is ik_residual below.
"""

import sys

import numpy as np

from fk_lite6 import fk_from_zeta
from lab_config import (IK_CHECK_JOINTS_DEG, IK_FD_EPS_DEG, JOINT_LIMITS_DEG,
                        TARGET_POSE, build_zeta_nominal)
from reporting import banner, field, progress
from so3 import rotation_vector
from units import euler_deg_from_matrix, transform_from_pose_deg

# The DH table the FK runs on, nominal.
ZETA_NOMINAL = build_zeta_nominal(6)


# =============================================================================
# THE RESIDUAL  --  this is the one function you write, Task 1(b)
# =============================================================================

def ik_residual(q_deg, T_target):
    """The 6-row residual e(q) = [position error in mm; orientation error in deg].

    q_deg is the joint vector in degrees, T_target the 4x4 flange target.
    """
    T_cur = fk_from_zeta(np.deg2rad(q_deg), ZETA_NOMINAL)

    # --- 1. Position: where the arm is, minus where you want it, in mm ------
    e_pos = <TO_FILL>

    # --- 2. Orientation: the rotation that separates the two orientations ---
    R_err = <TO_FILL>
    e_ori = rotation_vector(R_err)

    return np.concatenate([e_pos, e_ori])


# =============================================================================
# THE SOLVER  --  read it, do not rewrite it
# =============================================================================

def _project_limits(q, lb, ub):
    """Clip into the box, then nudge any joint sitting exactly on a bound inward."""
    qc = np.clip(q, lb, ub)
    at_lb = qc == lb
    at_ub = qc == ub
    if at_lb.any():
        qc[at_lb] = lb[at_lb] + 1e-6 * (ub[at_lb] - lb[at_lb])
    if at_ub.any():
        qc[at_ub] = ub[at_ub] - 1e-6 * (ub[at_ub] - lb[at_ub])
    return qc


def solve_ik(pose_deg, q0_deg=None, joint_limits_deg=None, *, max_iter=200,
             tol=1e-8, lam0=1e-3, verbose=True):
    """Target pose [x y z r p y] in mm and degrees -> dict of the answer.

    LM on ik_residual; joint_limits_deg, if given, projects every step into
    that box, so a start outside it is pulled in before the first iteration.
    """
    pose_deg = np.asarray(pose_deg, dtype=float)
    if pose_deg.shape != (6,):
        raise ValueError("solve_ik expects [x y z roll pitch yaw], got shape %s"
                         % (pose_deg.shape,))
    T_target = transform_from_pose_deg(pose_deg)

    if q0_deg is None:
        q = np.zeros(6)
    else:
        q = np.asarray(q0_deg, dtype=float).copy()
        if q.shape != (6,):
            raise ValueError("solve_ik expects 6 joint angles, got shape %s" % (q.shape,))

    use_limits = joint_limits_deg is not None
    if use_limits:
        lim = np.atleast_2d(joint_limits_deg)
        if lim.shape != (6, 2):
            raise ValueError("joint_limits_deg must be shape (6,2)")
        lb, ub = lim[:, 0], lim[:, 1]
        q = _project_limits(q, lb, ub)

    def cost(x):
        e = ik_residual(x, T_target)
        return 0.5 * float(e @ e), e

    def fd_jacobian(x):
        J = np.empty((6, 6))
        for j in range(6):
            xp = x.copy()
            xm = x.copy()
            xp[j] += IK_FD_EPS_DEG
            xm[j] -= IK_FD_EPS_DEG
            J[:, j] = (ik_residual(xp, T_target) -
                       ik_residual(xm, T_target)) / (2.0 * IK_FD_EPS_DEG)
        return J

    c_old, e_old = cost(q)
    if verbose:
        progress("start q(deg): " + " ".join("%7.2f" % v for v in q))
        progress("joint box: %s" % ("on" if use_limits else "off"))
        progress("  %4s  %12s  %12s  %10s  %10s"
                 % ("it", "lambda", "||dx||", "pos(mm)", "ori(deg)"))

    lam = lam0
    converged = False
    prev_c = c_old
    it = 0
    for it in range(1, max_iter + 1):
        # Damped least squares through the SVD, so a rank-deficient J at a
        # singularity still gives a finite step.
        U, s, Vt = np.linalg.svd(fd_jacobian(q), full_matrices=False)
        dx = -Vt.T @ ((s / (s ** 2 + lam)) * (U.T @ e_old))
        qt = _project_limits(q + dx, lb, ub) if use_limits else q + dx

        c_new, e_new = cost(qt)
        if c_new < c_old:
            q, e_old, c_old = qt, e_new, c_new
            # The same damping schedule as the calibration's lm_optimizer: the
            # bigger the cut in cost, the harder the damping is released.
            drop = (prev_c - c_new) / (abs(prev_c) + 1e-15)
            lam_factor = 0.1 if drop > 0.1 else (0.3 if drop > 0.01 else 0.5)
            lam = max(lam * lam_factor, 1e-12)
        else:
            lam *= 3.0
            if lam > 1e14:
                if verbose:
                    progress("lambda saturated at %.0e: the residual cannot be"
                             " reduced further from here" % lam)
                break

        if verbose:
            progress("  %4d  %12.2e  %12.2e  %10.4f  %10.4f"
                     % (it, lam, np.linalg.norm(dx),
                        np.linalg.norm(e_old[:3]), np.linalg.norm(e_old[3:])))

        if np.linalg.norm(dx) < tol:
            converged = True
            if verbose:
                progress("converged (||dx|| < tol) at iteration %d" % it)
            break
        prev_c = c_old

    e_final = ik_residual(q, T_target)
    result = {"q_deg": q,
              "success": converged,
              "iters": it,
              "residual_norm": float(np.linalg.norm(e_final)),
              "pos_err_mm": float(np.linalg.norm(e_final[:3])),
              "ori_err_deg": float(np.linalg.norm(e_final[3:])),
              "T": fk_from_zeta(np.deg2rad(q), ZETA_NOMINAL)}

    if verbose:
        field("result", "q(deg) " + " ".join("%7.2f" % v for v in q))
        field("error", "pos=%.4f mm  ori=%.4f deg  ||e||=%.4g"
              % (result['pos_err_mm'], result['ori_err_deg'],
                 result['residual_norm']))
        field("status", "CONVERGED" if converged else "NOT CONVERGED")

    return result


if __name__ == "__main__":
    if len(sys.argv) in (7, 13):
        # python3 ik_lite6.py X Y Z ROLL PITCH YAW [q1 ... q6]  (mm, degrees)
        # The six optional starting angles make one Table 1 guess one command.
        argv = [float(v) for v in sys.argv[1:]]
        pose = np.array(argv[:6])
        q0 = np.array(argv[6:]) if len(argv) > 6 else None
        banner("LITE 6 NUMERICAL INVERSE KINEMATICS")
        field("target", "pos=%8.3f %8.3f %8.3f mm   rpy=%7.3f %7.3f %7.3f deg"
              % (*pose[:3], *euler_deg_from_matrix(transform_from_pose_deg(pose)[:3, :3])))
        if q0 is None:
            progress("from rest, joints free; append q1..q6 to solve inside the box.")
        res = solve_ik(pose, q0_deg=q0,
                       joint_limits_deg=None if q0 is None else JOINT_LIMITS_DEG)
        # FK of the answer, back against the target.
        field("round trip", "%.2e mm from the target"
              % np.linalg.norm(res["T"][:3, 3] - pose[:3]))
        if not res["success"]:
            field("settled at", "pos %8.3f %8.3f %8.3f mm" % tuple(res["T"][:3, 3]))
            progress("NOT CONVERGED: the flange settled away from the target.")
    else:
        # Task 1(c): target built from a joint set with the FK, residual stood
        # at those same readings. A correct residual is zero there.
        q_check = np.array(IK_CHECK_JOINTS_DEG, dtype=float)
        T_target = fk_from_zeta(np.deg2rad(q_check), ZETA_NOMINAL)
        e_zero = ik_residual(q_check, T_target)

        # Verification
        T_target_v = fk_from_zeta(np.deg2rad(q_check+10), ZETA_NOMINAL)
        e_verification = ik_residual(q_check, T_target_v)


        banner("IK RESIDUAL SELF-CHECK")
        field("q (deg)", " ".join("%7.2f" % v for v in q_check))
        field("target", "flange at %8.3f %8.3f %8.3f mm" % tuple(T_target[:3, 3]))
        field("e (mm, deg)", "[" + "  ".join("%9.4f" % v for v in e_zero) + "]")
        field("||e||", "%.2e  (must be zero)" % np.linalg.norm(e_zero))

        banner("IK VERIFICATION SELF-CHECK")
        field("q (deg)", " ".join("%7.2f" % v for v in q_check))
        field("target", "flange at %8.3f %8.3f %8.3f mm" % tuple(T_target_v[:3, 3]))
        field("e (mm, deg)", "[" + "  ".join("%9.4f" % v for v in e_verification) + "]")
        field("||e||", "%.2e  (must be ~72.6mm)" % np.linalg.norm(e_verification))

        print()
        if np.linalg.norm(e_zero) < 1e-9 and abs(np.linalg.norm(e_verification) - 72.6)  < 1e-1:
            print("  All checks OK")
        else:
           print("  Check residual - it is wrong")
