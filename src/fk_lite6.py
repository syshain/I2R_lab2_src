"""Modified (Craig) D-H forward kinematics for the Lite 6, and the zeta layout.

Frames 0 = base, 6 = flange. zeta = [a | alpha | d | theta_off], n each, mm and radians.
"""

import numpy as np

import lab_config as C


def modified_dh_transform(a, alpha, d, theta):
    """Single-joint modified (Craig) D-H transform -> 4x4; a, d mm, angles rad."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st,       0,   a],
        [st*ca, ct*ca, -sa, -sa*d],
        [st*sa, ct*sa,  ca,  ca*d],
        [0.0,  0.0,     0.0, 1.0],
    ])


def fk_from_zeta(q, zeta, n=6):
    """Joint angles q (rad) + zeta -> T_0_6 (4x4, translation mm)."""
    a = zeta[0:n]
    alpha = zeta[n:2*n]
    d = zeta[2*n:3*n]
    theta_off = zeta[3*n:4*n]
    T = np.eye(4)

    for i in range(n):
        T = T @ modified_dh_transform(a[i], alpha[i], d[i], theta_off[i] + q[i])
    return T


def fk_from_zeta_batch(Qs, zeta, n=6):
    """fk_from_zeta over every row of Qs at once: (l, n) rad -> (l, 4, 4) mm.

    Same products in the same order, batched: about 9x on a 30-pose fit.
    """

    Qs = np.ascontiguousarray(Qs, dtype=float)
    l = Qs.shape[0]
    a = zeta[0:n]
    alpha = zeta[n:2 * n]
    d = zeta[2 * n:3 * n]
    theta_off = zeta[3 * n:4 * n]

    T = np.repeat(np.eye(4)[None, :, :], l, axis=0)
    for i in range(n):
        theta = theta_off[i] + Qs[:, i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        Ai = np.broadcast_to(modified_dh_transform(a[i], alpha[i], d[i], theta_off[i]),
                             (l, 4, 4)).copy()
        # Only the four angle-dependent entries change; the rest came from the
        # zero-angle template above.
        Ai[:, 0, 0] = ct
        Ai[:, 0, 1] = -st
        Ai[:, 1, 0] = st * ca
        Ai[:, 1, 1] = ct * ca
        Ai[:, 2, 0] = st * sa
        Ai[:, 2, 1] = ct * sa
        T = T @ Ai
    return T


# -----------------------------------------------------------------------------
# zeta naming, locking and printing -- the layout is defined here, so is its vocabulary
# -----------------------------------------------------------------------------

def dh_param_names(n=6):
    """zeta parameter names in vector order: a1..an, alpha1..n, d1..n, theta1..n."""
    return ([f'a{i + 1}' for i in range(n)] + [f'alpha{i + 1}' for i in range(n)] +
            [f'd{i + 1}' for i in range(n)] + [f'theta{i + 1}' for i in range(n)])


def locked_indices(names=None, n=6):
    """zeta indices held at nominal, ascending; names default to LOCKED_TABLE_NAMES."""
    if names is None:
        names = C.LOCKED_TABLE_NAMES
    idx = [dh_param_names(n).index(nm) for nm in names]
    return np.array(sorted(idx), dtype=int)


# Longest prefix first: 'alpha1' also starts with 'a'.
DH_BLOCKS = (('alpha', 'deg'), ('theta', 'deg'), ('a', 'mm'), ('d', 'mm'))


def unit_of(name):
    """Unit of a zeta parameter: millimetres for a and d, degrees for the angles."""
    for prefix, unit in DH_BLOCKS:
        if name.startswith(prefix):
            return unit
    raise ValueError("not a zeta parameter name: %r" % (name,))
