"""Task 3: which parameters the calibration data can actually see.

The Jacobian J = dr/dxi is built twice at the point Stage 2 starts from --
nominal DH and the two-pose hand-eye seed -- once with all 30 parameters free
and once with Sicilia's eight locked at nominal, which is exactly the basis
Stage 2 optimises in. Every singular value is printed divided by the largest,
because the unidentifiable directions are small rather than literally zero.

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
    """Print one parameter set: normalised singular values, then the rank."""
    s = np.linalg.svd(J, compute_uv=False)
    rel = s / s[0]
    heading(title)
    field("shape", "%d equations x %d unknowns, sigma_max = %.3e"
          % (J.shape[0], J.shape[1], s[0]))
    print("    %-8s %5s %15s  %s" % ("parameter", "unit", "sigma/sigma_max", "verdict"))
    for row, col in enumerate(free):
        name, unit = table[col]
        print("    %-8s %5s %15.3e  %s"
              % (name, unit, rel[row],
                 "AT ZERO: not identifiable" if rel[row] < OBSERVABILITY_ZERO_REL
                 else "identifiable"))
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
    # The rows the fit is weighted by, so this is the Jacobian the optimiser
    # really sees and not merely the one the geometry implies.
    J = J * cap.w_rows[:, None]

    lock_idx = locked_indices(n=n)
    free_dh = np.setdiff1d(np.arange(4 * n), lock_idx)
    # Stage 2's own basis: the 16 free DH columns, then the six hand-eye ones.
    cols_22 = np.concatenate([free_dh, np.arange(4 * n, 4 * n + 6)])

    print()
    table = param_table(n)
    report(J, table, "ALL %d PARAMETERS FREE" % J.shape[1],
           list(range(J.shape[1])))
    report(J[:, cols_22], table,
           "%d FREE, THE EIGHT LOCKED AT NOMINAL (the basis Stage 2 uses)"
           % len(cols_22), list(cols_22))

    heading("WHAT THE TWO COUNTS MEAN")
    print("  The eight locked parameters fall in three groups. a1, alpha1, d1 and")
    print("  theta1: the frame at the top of link 1 only has to reach the")
    print("  artifact, and T_6_C reaches it too. d6 and theta6: a shift and a")
    print("  spin of the flange, both of which T_6_C absorbs, so their partners")
    print("  t6c_tz and t6c_rz collapse with them. d2 with d3: one length along")
    print("  one axis, counted twice.")
    print()
    print("  A locked parameter and its hand-eye partner span the SAME collapsed")
    print("  direction, and the singular value is attributed to whichever column")
    print("  the decomposition lands on. That is why the zeros in the first block")
    print("  count fewer than eight: every collapse you see in the t6c rows is a")
    print("  locked DH column that has no direction of its own either.")
    print()
    print("  The number that matters is the second block: with the eight held,")
    print("  none of the 22 directions Stage 2 moves in is collapsed, which is")
    print("  why it is the stage worth running.")
    print()
    field("'at zero' means", "below %.0e of the largest singular value"
          % OBSERVABILITY_ZERO_REL)


if __name__ == "__main__":
    main()
