"""Run the three calibration stages on one capture, compare them, save them.

Reads data/calibration_data.npy; no robot connection, so a capture re-runs freely.
"""

import numpy as np

import lab_config as C
from calibration_io import (load_calibration_data, save_results_json,
                            save_results_npy, save_text)
from calibration_metrics import (compute_T0W_stats, fk_vs_reference_rmse,
                                 t6c_difference)
from reporting import (banner, consistency_lines, field, format_results_txt,
                       heading, print_comparison_table, print_dh_table,
                       print_stats_block, progress)
from run_stages import TITLES, run_stage1, run_stage2, run_stage_f


def main():
    """Fit all three stages on the stored capture; print them and write the results."""
    cap = load_calibration_data()
    Q, TCW_list, n, zeta, w_rows = cap.Q, cap.TCW_list, cap.n, cap.zeta_nominal, cap.w_rows

    banner("LITE 6 HAND-EYE AND KINEMATIC CALIBRATION")
    progress("loaded %s" % cap.path)
    progress("%d poses, %d joints, PnP reprojection %.2f px mean"
             % (len(Q), n, cap.reproj_px.mean() if len(cap.reproj_px) else float('nan')))
    progress("no robot connection is used; every stage here runs offline")
    print()

    # Baseline: nominal DH, hand-eye taken as the identity, nothing fitted.
    baseline_stats = compute_T0W_stats(Q, TCW_list, np.concatenate([zeta, np.zeros(6)]), n)
    print_stats_block("BASELINE: nominal DH, T_6_C = I, nothing fitted", baseline_stats)

    fk_ref = None
    if cap.T_0_6_controller is not None:
        fk_ref = fk_vs_reference_rmse(Q, cap.T_0_6_controller, zeta, n)
        heading("NOMINAL-DH FK VS CONTROLLER T_0_6")
        field("RMSE", "%.3f mm" % fk_ref['rmse_mm'])
        field("pose-dep", "%.3f mm" % fk_ref['pos_std_mm'])
        field("orientation", "%.2f deg mean, %.2f deg max"
              % (fk_ref['ori_mean_deg'], fk_ref['ori_max_deg']))
        print()

    banner(TITLES['stage1'])
    s1 = run_stage1(Q, TCW_list, zeta, w_rows, n)
    print_stats_block("Stage 1, over the poses it was fitted on", s1.stats)

    sf = None
    if cap.T_0_6_controller is None:
        progress("STAGE F NOT RUN: this capture stored no controller flange poses.")
        progress("Re-collect with the current collect_data.py to fit against them.")
    else:
        banner(TITLES['stagef'])
        sf = run_stage_f(cap.T_0_6_controller, Q, TCW_list, zeta, w_rows, n)
        print_stats_block("Stage F scored through the controller FK",
                          sf.extra['stats_controller'])
        print_stats_block("Stage F T_6_C scored through the nominal-DH FK",
                          sf.stats)

    banner(TITLES['stage2'])
    s2 = run_stage2(Q, TCW_list, zeta, w_rows, n, t6c_seed=s1.t6c)
    print_stats_block("Stage 2, over the poses it was fitted on", s2.stats)
    print_dh_table(s2.zeta, n, compare_to=zeta, free=set(s2.free_idx.tolist()))

    rows = [("Baseline", baseline_stats), ("Stage 1 hand-eye", s1.stats)]
    if sf is not None:
        rows.append(("Stage F controller FK", sf.extra['stats_controller']))
    rows.append(("Stage 2 DH + HE", s2.stats))
    print_comparison_table(rows, "ALL CALIBRATIONS COMPARED")

    return save_results(cap, baseline_stats, fk_ref, s1, s2, sf, rows)


def save_results(cap, baseline_stats, fk_ref, s1, s2, sf, rows):
    """Write the results three ways: .npy for code, .json and .txt for reading."""
    n = cap.n
    results = {
        'zeta_nominal': cap.zeta_nominal,
        'n_poses': len(cap.Q),
        'baseline_stats': baseline_stats,
        # t6c_stage1 / zeta_stage2 / t6c_stage2 / t6c_stagef are the keys
        # validate_calibration.py builds its candidates from.
        't6c_stage1': s1.t6c, 'stage1_stats': s1.stats,
        'stage1_iters': s1.iters, 'stage1_converged': s1.converged,
        'stage1_status': s1.status,
        't6c_stage2': s2.t6c, 'zeta_stage2': s2.zeta, 'stage2_stats': s2.stats,
        'stage2_iters': s2.iters, 'stage2_converged': s2.converged,
        'stage2_status': s2.status,
        'stage2_free_idx': s2.free_idx, 'stage2_locked_idx': s2.extra['lock_idx'],
    }
    if fk_ref is not None:
        results['fk_vs_controller'] = fk_ref
    if sf is not None:
        results.update({'t6c_stagef': sf.t6c,
                        'stagef_stats': sf.stats,
                        'stagef_stats_controller': sf.extra['stats_controller'],
                        'stagef_iters': sf.iters, 'stagef_converged': sf.converged,
                        'stagef_status': sf.status})
        for other, label in [(s1, 'stage1'), (s2, 'stage2')]:
            dt, dr = t6c_difference(sf.t6c, other.t6c)
            results['t6c_stagef_vs_%s_mm' % label] = dt
            results['t6c_stagef_vs_%s_deg' % label] = dr
    context = {'n_joints': n, 'n_poses': len(cap.Q), 'data_file': cap.path,
               'zeta_nominal': cap.zeta_nominal, 'fk_vs_controller': fk_ref,
               'baseline_stats': baseline_stats, 'summary_rows': rows,
               'weight_position': C.WEIGHT_POSITION,
               'weight_orientation': C.WEIGHT_ORIENTATION}
    stages = [
        {'title': TITLES['stage1'], 'stats': s1.stats, 't6c': s1.t6c,
         'iters': s1.iters, 'converged': s1.converged, 'status': s1.status},
    ]
    if sf is not None:
        stages.append({'title': TITLES['stagef'],
                       'stats': sf.extra['stats_controller'], 't6c': sf.t6c,
                       'iters': sf.iters, 'converged': sf.converged,
                       'status': sf.status,
                       'extra_lines': [
                           ["-- The same T_6_C scored through the nominal-DH FK --"]
                           + consistency_lines(sf.stats)]})
    stages.append(
        {'title': TITLES['stage2'], 'stats': s2.stats, 't6c': s2.t6c,
         'zeta': s2.zeta, 'free_idx': set(s2.free_idx.tolist()),
         'iters': s2.iters, 'converged': s2.converged, 'status': s2.status})

    report = format_results_txt(context, stages)
    for path, write in [(C.data_path(C.RESULTS_NPY_NAME),
                         lambda p: save_results_npy(p, results)),
                        (C.data_path(C.RESULTS_JSON_NAME),
                         lambda p: save_results_json(p, results, n)),
                        (C.data_path(C.RESULTS_TXT_NAME),
                         lambda p: save_text(p, report))]:
        write(path)
        progress("wrote %s" % path)
    return results


if __name__ == '__main__':
    main()
