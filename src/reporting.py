"""All console output and the text report, in one grammar.

The grammar itself, which is what the reader needs, is spelled out below.
"""

#     .. message       process noise: loading, capturing, iterating. Ignore it.
#     == TITLE ==      a stage or section.
#     -- Heading       a block of results within a stage.
#       label   value  a result. Anything in this shape is a number to read off.

import numpy as np

from calibration_metrics import quality_grade
from fk_lite6 import dh_param_names, unit_of
from units import euler_deg_from_matrix
from so3 import t6c_from_vec


def banner(title):
    """Print a section title between double rules."""
    print("=" * 64)
    print(" " + title)
    print("=" * 64)


def heading(title):
    """Print a sub-block title."""
    print("  -- %s" % title)


def progress(message):
    """Print process noise, visually set apart from results."""
    print("  .. %s" % message)


def field(label, value, note=''):
    """Print one result line: aligned label, formatted value, optional note."""
    text = "    %-11s %s" % (label, value)
    if note:
        text += "   %s" % note
    print(text)


def print_stats_block(label, stats):
    """Print one consistency measurement: per-axis spread, worst axis, worst error."""
    heading(label)
    field("Pos std", "X=%.3f  Y=%.3f  Z=%.3f mm" % tuple(stats['pos_axis_std_mm']))
    field("s_max", "%.3f mm (%s axis)" % (stats['s_max_mm'], stats['s_max_axis']),
          quality_grade(stats['s_max_mm']))
    field("Trans max", "%.3f mm" % stats['trans_max_mm'])
    field("Ori", "mean %.3f deg  max %.3f deg"
          % (stats['ori_mean_deg'], stats['ori_max_deg']))
    print()


def comparison_lines(rows, title="COMPARISON TABLE", width=None):
    """The ranking to read off: rows of (label, stats), best s_max first."""
    width = max(width or 0, 15, max(len(label) for label, _ in rows))
    lines = [title, "",
             "%-*s %12s %12s %12s %12s"
             % (width, "Method", "s_max (mm)", "TransMax", "OriMax (deg)", "Grade")]
    for label, stats in sorted(rows, key=lambda r: r[1]['s_max_mm']):
        lines.append("%-*s %12.3f %12.3f %12.3f %12s"
                     % (width, label, stats['s_max_mm'], stats['trans_max_mm'],
                        stats['ori_max_deg'], quality_grade(stats['s_max_mm'])))
    lines.append("")
    return lines


def print_comparison_table(rows, title="FINAL COMPARISON"):
    """Print the stage ranking, best s_max first."""
    for line in comparison_lines(rows, title):
        print(line)


def t6c_block(t6c_vec, label):
    """The lines describing one fitted T_6_C, shared by the console and the report."""
    T = t6c_from_vec(t6c_vec)
    rpy = euler_deg_from_matrix(T[:3, :3])
    return [
        "-- %s" % label,
        "  position     x=%.3f  y=%.3f  z=%.3f mm" % tuple(T[:3, 3]),
        "  orientation  roll=%.3f  pitch=%.3f  yaw=%.3f deg" % tuple(rpy),
        "  raw vector   [" + "  ".join("%9.4f" % v for v in t6c_vec) + "]",
        "",
    ]


def consistency_lines(stats):
    """The consistency block of one stage, as report lines."""
    return [
        "  Pos std       : X=%.4f  Y=%.4f  Z=%.4f mm" % tuple(stats['pos_axis_std_mm']),
        "  s_max         : %.4f mm (%s axis)  [%s]"
        % (stats['s_max_mm'], stats['s_max_axis'], quality_grade(stats['s_max_mm'])),
        "  Trans max     : %.4f mm" % stats['trans_max_mm'],
        "  Orientation   : mean %.4f deg, max %.4f deg"
        % (stats['ori_mean_deg'], stats['ori_max_deg']),
        "",
    ]


def optimiser_lines(iters, converged, status):
    """How the optimiser stopped, as report lines."""
    return ["  Iterations    : %d  (%s%s)"
            % (iters, status, "" if converged else ", NOT converged"), ""]


def dh_table_body_lines(zeta, n, compare_to=None, free=None):
    """DH rows alone: a column header then one line per parameter, no title."""
    names = dh_param_names(n)
    if free is None:
        free = set(range(len(names)))
    lines = []
    if compare_to is None:
        lines.append("  %-8s %6s %12s" % ("param", "unit", "value"))
        for name, value in zip(names, zeta):
            lines.append("  %-8s %6s %12.5f" % (name, unit_of(name), value))
    else:
        lines.append("  %-8s %6s %12s %12s %12s"
                     % ("param", "unit", "nominal", "calibrated", "delta"))
        for idx, name in enumerate(names):
            unit = unit_of(name)
            conv = (lambda v: np.rad2deg(v)) if unit == 'deg' else (lambda v: v)
            mark = "" if idx in free else "   held at nominal"
            lines.append("  %-8s %6s %12.5f %12.5f %+12.5f%s"
                        % (name, unit, conv(compare_to[idx]), conv(zeta[idx]),
                           conv(zeta[idx] - compare_to[idx]), mark))
    return lines


def dh_table_lines(zeta, n, compare_to=None, free=None,
                   title="-- Calibrated DH parameters (angles in degrees) --"):
    """The DH table as report lines, framed by a title and a closing blank."""
    return [title] + dh_table_body_lines(zeta, n, compare_to, free) + [""]


def print_dh_table(zeta, n, title="DH parameters (angles in degrees)", **kwargs):
    """Print the DH table under a heading, indented like field(), then a blank."""
    heading(title)
    for line in dh_table_body_lines(zeta, n, **kwargs):
        print("  " + line)
    print()


def validation_gain_lines(candidates, width=26):
    """Fitted-vs-hold-out rows as report lines; positive means worse than it looked."""
    return ["  %-*s fitted %8.3f mm   hold-out %8.3f mm   (%+7.3f)"
            % (width, cand['label'], cand['fitted_s_max_mm'],
               cand['stats']['s_max_mm'],
               cand['stats']['s_max_mm'] - cand['fitted_s_max_mm'])
            for cand in candidates]


def format_results_txt(context, stages):
    """The whole calibration report as one string.

    context holds the run-level figures; stages is one dict per stage.
    """

    n = context['n_joints']
    fk = context['fk_vs_controller']
    L = [
        "=" * 64,
        "LITE 6 HAND-EYE AND KINEMATIC CALIBRATION RESULTS",
        "=" * 64,
        "",
        "Data            : %s" % context['data_file'],
        "Poses           : %d   Joint groups: %d" % (context['n_poses'], n),
        "Weights         : position=%.2f  orientation=%.2f"
        % (context['weight_position'], context['weight_orientation']),
        "",
    ]
    if fk is not None:
        L += [
            "-- Model disagreement (a difference of models, not an error) --",
            "  Our nominal-DH FK vs the controller's own T_0_6, over these poses:",
            "  RMSE %.3f mm, pose-dependent part %.3f mm, orientation %.2f deg mean"
            " / %.2f deg max" % (fk['rmse_mm'], fk['pos_std_mm'],
                                 fk['ori_mean_deg'], fk['ori_max_deg']),
            "  Any constant part of that is absorbed by the hand-eye fit, so s_max",
            "  would not move if it were removed. The pose-dependent part is what",
            "  Stages 2 and F are able to remove.",
            "",
        ]
    L += [
        "-- Baseline: nominal DH, T_6_C = I, nothing fitted --",
    ]
    L += consistency_lines(context['baseline_stats'])
    L += [""]

    for st in stages:
        L += ["=" * 64, st['title'], "=" * 64, ""]
        L += ["-- Consistency over the %d poses used to fit it --" % context['n_poses']]
        L += consistency_lines(st['stats'])
        if 'status' in st:
            L += optimiser_lines(st['iters'], st['converged'], st['status'])
        L += t6c_block(st['t6c'], "Fitted T_6_C")
        if st.get('zeta') is not None:
            L += dh_table_lines(st['zeta'], n, compare_to=context['zeta_nominal'],
                                free=st.get('free_idx'))
        for block in st.get('extra_lines', []):
            L += block

    L += comparison_lines(context['summary_rows'])
    gains = context.get('gains') or []
    if gains:
        s1_row = dict(context['summary_rows']).get('Stage 1 hand-eye')
        L += ["-- Improvement over Stage 1 (handout section 6.6) --"]
        for label, pct in gains:
            L.append("  %-24s %+7.1f %%   (against Stage 1 s_max %.3f mm)"
                     % (label, pct, s1_row['s_max_mm']))
        L += [""]
    L += [
        "s_max is the largest per-axis standard deviation of T_0_W over the poses",
        "used to fit it. It measures self-consistency, not accuracy: a constant error",
        "anywhere in the chain is invisible to it, because a constant is exactly what",
        "fitting T_6_C absorbs. Stage F has a sharper check available -- its own",
        "forward model against the controller's T_0_6 at the same poses, a difference",
        "that cannot hide inside T_6_C -- and both are tested on relocated markers by",
        "validate_calibration.py, which is the test that measures accuracy.",
        "",
    ]
    return "\n".join(L)


def format_validation_txt(context, candidates):
    """The relocated-artifact report as one string; one block per candidate."""
    n = context['n_joints']
    L = [
        "=" * 64,
        "LITE 6 CALIBRATION VALIDATION - RELOCATED ARTIFACT",
        "=" * 64,
        "",
        "Calibration read from : %s" % context['results_file'],
        "Hold-out capture      : %s" % context['validation_file'],
        "Hold-out poses        : %d   Joint groups: %d" % (context['n_poses'], n),
        "",
        "Nothing is fitted here. Every calibration below is frozen and scored on",
        "poses it never saw, with the artifact in a place it was not in when they",
        "were fitted.",
        "",
    ]

    moved = context.get('displacement')
    if moved is not None:
        L += [
            "-- The artifact moved --",
            "  W was (%9.2f, %9.2f, %9.2f) mm, now (%9.2f, %9.2f, %9.2f) mm"
            % (*moved['before_mm'], *moved['after_mm']),
            "  %.1f mm apart, chained through the controller's own T_0_6 and %s,"
            % (moved['moved_mm'], context.get('move_t6c', 'a fitted T_6_C')),
            "  so no DH table takes any part in the measurement.",
            "",
        ]

    for cand in candidates:
        L += ["=" * 64, cand['label'], "=" * 64, ""]
        L += ["-- Consistency over the %d relocated poses --" % context['n_poses']]
        L += consistency_lines(cand['stats'])
        if cand.get('t6c') is not None:
            L += t6c_block(cand['t6c'], "T_6_C exactly as fitted")
        if cand.get('zeta') is not None:
            L += dh_table_lines(cand['zeta'], n, compare_to=context['zeta_nominal'],
                                free=cand.get('free'))

    L += comparison_lines([(c['label'], c['stats']) for c in candidates],
                          "VALIDATION COMPARISON (best s_max first)")

    scored = [c for c in candidates if c.get('fitted_s_max_mm') is not None]
    if scored:
        L += ["", "-- Same calibrations, measured twice --"]
        L += validation_gain_lines(scored)
        L += ["",
              "The fitted column is s_max over the poses a stage was optimised on,",
              "the hold-out column over poses of a moved artifact. A stage that got",
              "its low s_max by absorbing the first artifact's noise rather than by",
              "modelling the arm gets a large gap here; a stage that genuinely",
              "improved the model keeps its lead. Hold-out s_max is still",
              "self-consistency, not accuracy: relocating the artifact removes the",
              "evidence a fit could overfit, it does not add a ground truth.",
              ""]
    return "\n".join(L)
