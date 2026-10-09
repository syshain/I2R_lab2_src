"""Levenberg-Marquardt least squares: row weighting, L2 regularisation, SVD mask.

Blind to what the parameters mean; every tunable comes from lab_config; silent unless verbose.
"""

# SIGN CONTRACT -- the one rule the solver cannot check for you.
#
#     residual_fn must return J = -dr/dxi, NOT +dr/dxi.
#
# The augmented system below is assembled with +rw on the right (textbook LM
# uses -rw) and the update is x <- x + dx, so the two minus signs cancel and
# the loop walks downhill only if J carries the explicit one. Hand it a plain
# +dr/dxi and every trial step points uphill: each is rejected, lam triples,
# and it exits on iteration 1 with zero parameter movement. That reads like an
# ill-conditioned problem, not a sign error, which is how this very fit
# came to stall once before.
#
# calibration_model satisfies it the same way everywhere: the residual is
# built as r = ref - cur and the finite difference is taken on (cur - ref).

import numpy as np

import lab_config as C
from reporting import progress


def levenberg_marquardt(residual_fn, xi0, w_rows, *, max_iter=C.LM_MAX_ITER,
                        tol=C.LM_TOL, lam0=C.LM_LAM0, reg_weight=0.0,
                        svd_thresh=0.0, verbose=True):
    """Minimise 0.5*||w*r(xi)||^2 + 0.5*reg_weight*||xi - xi0||^2.

    residual_fn -> (r, J) with J = -dr/dxi; returns xi, iters, converged, status.
    """

    x = xi0.copy()
    lam = lam0
    n_params = len(x)
    l = len(w_rows) // 6

    def _mean_errors(rw):
        """Mean per-pose position error (mm) and orientation error (deg)."""
        pos = [np.linalg.norm(rw[i * 6:i * 6 + 3]) / w_rows[0] for i in range(l)]
        ori = [np.linalg.norm(rw[i * 6 + 3:i * 6 + 6]) / w_rows[3] * 180 / np.pi
               for i in range(l)]
        return np.mean(pos), np.mean(ori)

    def total_cost(xk, r_cached=None):
        """Weighted data cost plus the L2 pull toward xi0."""
        r = (np.asarray(residual_fn(xk)[0], dtype=float).reshape(-1)
             if r_cached is None else r_cached)
        rw = r * w_rows
        cost = 0.5 * np.dot(rw, rw)
        if reg_weight > 0:
            dx0 = xk - xi0
            cost += 0.5 * reg_weight * np.dot(dx0, dx0)
        return cost

    # One evaluation at the start supplies both r and J for iteration 1.
    r_old, J0 = residual_fn(x)
    r_old = np.asarray(r_old, dtype=float).reshape(-1)
    rw_old = r_old * w_rows
    c_old = total_cost(x, r_cached=r_old)

    if verbose:
        pe, oe = _mean_errors(rw_old)
        progress("optimiser: %d poses, %d params, %d equations"
                 % (l, n_params, l * 6))
        progress("weights: pos=%.0f  ori=%.0f  reg=%.0e  svd=%.0e" %
                 (w_rows[0], w_rows[3], reg_weight, svd_thresh))
        progress("start: pos=%.3f mm  ori=%.3f deg" % (pe, oe))
        progress("  %4s  %12s  %12s  %10s  %10s" %
                 ("it", "lambda", "||dx||", "pos(mm)", "ori(deg)"))

    # Observability mask from the starting Jacobian
    obs_mask = np.ones(n_params, dtype=bool)
    if svd_thresh > 0:
        JW0 = J0 * w_rows[:, np.newaxis]
        _, s0, _ = np.linalg.svd(JW0, full_matrices=False)
        obs_mask = s0 > svd_thresh * s0[0]
        n_obs = int(obs_mask.sum())
        if verbose and n_obs < n_params:
            progress("observable: %d/%d (freezing %d)"
                     % (n_obs, n_params, n_params - n_obs))

    converged = False
    status = 'max-iter'
    prev_cost = c_old
    prev_dx_norm = None
    slow_count = 0

    # residual_fn's second output is a full finite-difference sweep, the most
    # expensive call here, and it only changes when x does. A rejected step
    # moves lam alone, so its Jacobian is still valid next iteration.
    # Anything that assigns to x must invalidate this.
    J_cached = J0

    for it in range(1, max_iter + 1):
        if J_cached is None:
            _, J = residual_fn(x)      # x has moved since the last sweep
        else:
            J = J_cached               # first iteration, or a step just rejected
        J_cached = None
        Jw = J * w_rows[:, np.newaxis]

        # Augmented system with +rw (pairs with J = -dr/dxi, see above):
        #   [Jw; sqrt(reg)*I] dx = [rw; sqrt(reg)*(xi0 - x)]
        if reg_weight > 0:
            Ja = np.vstack([Jw, np.sqrt(reg_weight) * np.eye(n_params)])
            ra = np.concatenate([rw_old, np.sqrt(reg_weight) * (xi0 - x)])
        else:
            Ja = Jw
            ra = rw_old

        U, s, Vt = np.linalg.svd(Ja, full_matrices=False)
        d = s / (s ** 2 + lam)
        if svd_thresh > 0:
            d[~obs_mask] = 0.0
        dx = Vt.T @ (d * (U.T @ ra))

        c_prev = c_old
        xt = x + dx
        # One sweep at the trial point gives both the new cost and, if the step
        # is taken, the new residual.
        r_trial = np.asarray(residual_fn(xt)[0], dtype=float).reshape(-1)
        cn = total_cost(xt, r_cached=r_trial)

        if cn < c_old:
            x = xt
            rw_old = r_trial * w_rows
            c_old = cn
            J_cached = None            # x moved: the cached Jacobian is stale
            improvement = (c_prev - cn) / (abs(c_prev) + 1e-15)
            if improvement > C.LM_STRONG_DROP:
                lam_factor = C.LM_LAM_DOWN_STRONG
            elif improvement > C.LM_MEDIUM_DROP:
                lam_factor = C.LM_LAM_DOWN_MEDIUM
            else:
                lam_factor = C.LM_LAM_DOWN_WEAK
            lam = max(lam * lam_factor, C.LM_LAM_FLOOR)
        else:
            J_cached = J               # x never moved, so the Jacobian stands
            lam *= C.LM_LAM_UP
            if lam > C.LM_LAM_CEILING:
                status = 'lam-saturated'
                if verbose:
                    print()
                    progress("lambda saturated at %.0e" % lam)
                break

        pe, oe = _mean_errors(rw_old)
        if verbose:
            progress("  %4d  %12.2e  %12.2e  %10.4f  %10.4f" %
                     (it, lam, np.linalg.norm(dx), pe, oe))

        dx_norm = np.linalg.norm(dx)

        if dx_norm < tol:
            converged = True
            status = 'tolerance'
            if verbose:
                print()
                progress("converged (||dx|| < tol) at iteration %d" % it)
            break

        rel_drop = (prev_cost - c_old) / (abs(prev_cost) + 1e-15)
        if rel_drop < C.LM_STALL_REL:
            status = 'stalled'
            if verbose:
                print()
                progress("stalled: cost drop < %.0e, iteration %d"
                       % (C.LM_STALL_REL, it))
            break
        prev_cost = c_old

        # Damping at its floor and ||dx|| barely changing: still running, gaining nothing.
        if lam <= C.LM_CRAWL_LAM and prev_dx_norm is not None:
            dx_change = abs(dx_norm - prev_dx_norm) / (prev_dx_norm + 1e-15)
            slow_count = slow_count + 1 if dx_change < C.LM_CRAWL_REL else 0
            if slow_count >= C.LM_CRAWL_COUNT:
                converged = True
                status = 'slow-crawl'
                if verbose:
                    print()
                    progress("converged (slow crawl) at iteration %d" % it)
                break
        prev_dx_norm = dx_norm

    if status == 'max-iter' and verbose:
        print()
        progress("max iterations reached (%d)" % max_iter)

    return {"xi": x, "converged": converged, "status": status, "iters": it}
