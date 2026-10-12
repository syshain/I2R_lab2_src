"""Singular values of the calibration Jacobian, for Table 6.

J = dr/dxi is built at the point Stage 2 starts from (nominal DH, two-pose
hand-eye seed), once with all 30 parameters free and once with the eight
locked at nominal. Each singular value is printed divided by the largest.

    python3 observability.py
"""

import numpy as np

from calibration_io import build_weight_vector, load_calibration_data
from calibration_model import compute_residuals_and_jacobian, init_T6C_two_pose
from fk_lite6 import dh_param_names, locked_indices, unit_of
from lab_config import N_DOF, OBSERVABILITY_ZERO_REL
from reporting import banner, field, heading


def param_table(n=N_DOF):
    """(name, unit) for the whole 4n + 6 vector, DH first then the hand-eye."""
    rows = [(name, unit_of(name)) for name in dh_param_names(n)]
    rows += [('t6c_tx', 'mm'), ('t6c_ty', 'mm'), ('t6c_tz', 'mm'),
             ('t6c_rx', 'rad'), ('t6c_ry', 'rad'), ('t6c_rz', 'rad')]
    return rows


def report(J, table, title, free):
    """
    Print one parameter set: normalised singular values, then the rank.
    """
    DIR_MIN = 0.15          # component large enough for the parameter to be named
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    rel = s / s[0]
    heading(title)
    field("shape", "%d equations x %d unknowns, sigma_max = %.3e"
          % (J.shape[0], J.shape[1], s[0]))
    print("    %-34s %15s  %s" % ("direction", "sigma/sigma_max", "verdict"))
    for row in range(len(rel)):
        at_zero = rel[row] < OBSERVABILITY_ZERO_REL
        # A collapsed direction is the one a pair has to read, so it is named in
        # full. The rest are named by their largest two, to keep the list short.
        order = np.argsort(-abs(Vt[row]))[:6 if at_zero else 2]
        names = [table[free[i]][0] for i in order if abs(Vt[row][i]) > DIR_MIN]
        print("    %-34s %15.3e  %s"
              % (", ".join(names) or table[free[order[0]]][0], rel[row],
                 "AT ZERO: not identifiable" if at_zero else "identifiable"))
    n_zero = int(np.sum(rel < OBSERVABILITY_ZERO_REL))
    field("rank", "%d of %d columns; %d collapsed below %.0e of sigma_max"
          % (len(rel) - n_zero, len(rel), n_zero, OBSERVABILITY_ZERO_REL))
    print()
    return rel

def main():
    banner("TASK 3 - OBSERVABILITY OF THE CALIBRATION PARAMETERS")

    cap = load_calibration_data()
    n = cap.n
    t6c_seed = init_T6C_two_pose(cap.Q, cap.TCW_list, cap.zeta_nominal, n)
    xi = np.concatenate([cap.zeta_nominal, t6c_seed])

    _, J = compute_residuals_and_jacobian(xi, cap.Q, cap.TCW_list, n=n)
    J = J * cap.w_rows[:, None]

    lock_idx = locked_indices(n=n)
    free_dh = np.setdiff1d(np.arange(4 * n), lock_idx)
    cols_22 = np.concatenate([free_dh, np.arange(4 * n, 4 * n + 6)])

    print()
    table = param_table(n)
    report(J, table, "ALL %d PARAMETERS FREE" % J.shape[1],
           list(range(J.shape[1])))
    report(J[:, cols_22], table,
           "%d FREE, EIGHT LOCKED AT NOMINAL (Stage 2's basis)" % len(cols_22),
           list(cols_22))

if __name__ == "__main__":
    main()
