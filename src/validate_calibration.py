"""Validate the saved calibrations against a relocated artifact; nothing is refitted.

Every calibration in calibration_results.npy is scored on poses it never saw.
"""

import sys

import numpy as np

import lab_config as C
from aruco_artifact_detector import ArucoArtifactDetector
from calibration_io import (load_camera_intrinsics, load_calibration_data,
                            load_results, load_validation_capture, save_capture,
                            save_results_npy, save_text)
from calibration_metrics import (artifact_displacement, compute_T0W_stats,
                                 compute_T0W_stats_from_flange)
from capture_view import capture_loop
from collect_data import record_pose
from reporting import (banner, field, format_validation_txt, heading,
                       print_comparison_table, print_stats_block, progress,
                       validation_gain_lines)
from robot import connect, disconnect


def capture_hold_out(n=C.N_DOF):
    """Relocate the artifact, drive the arm through new poses -> CaptureSet or None."""
    path = C.data_path(C.VALIDATION_DATA_NAME)
    zeta_nominal = C.build_zeta_nominal(n)

    K, dist = load_camera_intrinsics()
    detector = ArucoArtifactDetector(K=K, dist=dist)
    arm = connect()

    banner("Validation capture: relocated artifact")
    field("arm", C.ROBOT_IP)
    field("camera", "%d" % detector.camera_index)
    field("output", path)
    field("poses wanted", "%d to %d"
          % (C.VALIDATION_MIN_POSES, C.VALIDATION_MAX_POSES))
    progress("Move the artifact first, then leave it still. Nothing is refitted.")
    progress("Take the arm to its new poses in Manual Mode on the pendant:")
    progress("recording only reads the encoders, it never commands the arm.")
    progress("SPACE records a pose, ESC finishes.")

    records = []

    def on_capture(detection):
        """One accepted frame -> (ok, note shown on the preview)."""
        return record_pose(arm, detection, records, zeta_nominal, n)

    try:
        capture_loop(detector, on_capture, max_poses=C.VALIDATION_MAX_POSES,
                     window="Validation capture")
    finally:
        detector.close()
        disconnect(arm)

    if not save_capture(path, records, zeta_nominal, n,
                        min_poses=C.VALIDATION_MIN_POSES):
        return None
    return load_validation_capture(path)


def build_candidates(results, cap):
    """Every calibration the results file holds, as the rows to score.

    Each has a label and its model: 'zeta' + 't6c', or 'through': 'controller'.
    """

    zeta_nominal = results.get('zeta_nominal')
    if zeta_nominal is None:
        zeta_nominal = C.build_zeta_nominal(cap.n)

    candidates = [
        {'label': 'Baseline, nothing fitted', 'zeta': zeta_nominal,
         't6c': np.zeros(6), 'fitted_key': 'baseline_stats'},
        {'label': 'Stage 1 hand-eye', 'zeta': zeta_nominal,
         't6c': results['t6c_stage1'], 'fitted_key': 'stage1_stats'},
        {'label': 'Stage 2 DH + hand-eye', 'zeta': results['zeta_stage2'],
         't6c': results['t6c_stage2'], 'fitted_key': 'stage2_stats',
         'free': set(np.asarray(results['stage2_free_idx']).tolist())},
    ]

    # Stage F was fitted inside the controller's own kinematics, so it is scored
    # both ways: there, and through our nominal DH as a comparison.
    t6cf = results.get('t6c_stagef')
    if t6cf is not None and cap.T_0_6_controller is not None:
        candidates += [
            {'label': 'Stage F, controller FK', 't6c': t6cf,
             'through': 'controller', 'fitted_key': 'stagef_stats_controller'},
            {'label': 'Stage F, nominal DH FK', 'zeta': zeta_nominal,
             't6c': t6cf, 'fitted_key': 'stagef_stats'},
        ]
    return candidates


def score(candidate, cap):
    """One frozen calibration's consistency over a capture it never saw."""
    if candidate.get('through') == 'controller':
        return compute_T0W_stats_from_flange(cap.T_0_6_controller, cap.TCW_list,
                                             candidate['t6c'])
    return compute_T0W_stats(cap.Q, cap.TCW_list,
                             np.concatenate([candidate['zeta'], candidate['t6c']]),
                             cap.n)


def report_artifact_move(cap, t6c_named):
    """Print how far the artifact moved, or why that could not be checked."""
    if cap.T_0_6_controller is None:
        progress("the hold-out set stores no controller flange poses, so the move")
        progress("cannot be measured independently of the kinematics being scored")
        return None
    try:
        fitted = load_calibration_data()
    except FileNotFoundError:
        progress("the calibration capture is gone, so nothing can be compared to it")
        return None
    if fitted.T_0_6_controller is None:
        progress("the calibration capture stores no controller flange poses")
        return None

    moved = artifact_displacement(fitted.T_0_6_controller, fitted.TCW_list,
                                  cap.T_0_6_controller, cap.TCW_list,
                                  t6c_named[1])
    heading("The artifact moved")
    field("W before", "(%8.2f, %8.2f, %9.2f) mm" % tuple(moved['before_mm']))
    field("W after", "(%8.2f, %8.2f, %9.2f) mm" % tuple(moved['after_mm']))
    field("moved", "%.1f mm" % moved['moved_mm'],
          "controller T_0_6 and %s only, so no DH table takes part" % t6c_named[0])
    if moved['moved_mm'] < C.VALIDATION_MIN_MOVE_MM:
        field("WARNING", "that is barely a move",
              "relocate it %d mm or more and re-run; this tests little"
              % C.VALIDATION_MIN_MOVE_MM)
    print()
    return moved


def main(recapture=False):
    """Score every saved calibration on the relocated artifact and write the report."""
    results_file = C.data_path(C.RESULTS_NPY_NAME)
    results = load_results(results_file)

    cap = None if recapture else load_validation_capture()
    if cap is None:
        cap = capture_hold_out()
        if cap is None:
            banner("Validation not run")
            progress("too few poses, so nothing was scored and nothing was "
                     "overwritten")
            return None

    banner("VALIDATION ON A RELOCATED ARTIFACT")
    progress("calibrations read from %s" % results_file)
    progress("hold-out poses read from %s" % cap.path)
    field("hold-out poses", len(cap.Q))
    field("reproj mean", "%.2f px" % cap.reproj_px.mean())
    print()

    candidates = build_candidates(results, cap)
    for candidate in candidates:
        candidate['stats'] = score(candidate, cap)
        candidate['fitted_s_max_mm'] = (results.get(candidate.pop('fitted_key'))
                                        or {}).get('s_max_mm')

    # Any fitted T_6_C does for measuring the move; it is physical, so it stays in.
    t6c_for_move = (("Stage 1's T_6_C", results['t6c_stage1'])
                    if results.get('t6c_stage1') is not None
                    else ("the identity T_6_C", np.zeros(6)))
    moved = report_artifact_move(cap, t6c_for_move)

    for candidate in candidates:
        print_stats_block(candidate['label'], candidate['stats'])
    print_comparison_table([(c['label'], c['stats']) for c in candidates],
                           "VALIDATION COMPARISON (best s_max first)")

    scored = [c for c in candidates if c['fitted_s_max_mm'] is not None]
    if scored:
        heading("Same calibrations, measured twice")
        for line in validation_gain_lines(scored):
            print(line)
        print()

    context = {'results_file': results_file, 'validation_file': cap.path,
               'n_poses': len(cap.Q), 'n_joints': cap.n,
               'move_t6c': t6c_for_move[0],
               'zeta_nominal': results.get('zeta_nominal',
                                           C.build_zeta_nominal(cap.n)),
               'displacement': moved}
    save_text(C.data_path(C.VALIDATION_TXT_NAME),
              format_validation_txt(context, candidates))
    save_results_npy(C.data_path(C.VALIDATION_NPY_NAME), {
        'validation_file': cap.path, 'n_poses': len(cap.Q),
        'displacement': moved,
        's_max_mm': {c['label']: c['stats']['s_max_mm'] for c in candidates},
        'fitted_s_max_mm': {c['label']: c['fitted_s_max_mm']
                            for c in candidates}})
    for name in (C.VALIDATION_TXT_NAME, C.VALIDATION_NPY_NAME):
        progress("wrote %s" % C.data_path(name))
    return candidates


if __name__ == "__main__":
    main(recapture='--recapture' in sys.argv)
