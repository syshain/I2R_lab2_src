"""Reading capture data and writing results. Nothing here computes anything.

Every dataset and report lives in the folder named by lab_config.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

import lab_config as C
from fk_lite6 import dh_param_names, unit_of
from reporting import progress
from units import joints_to_radians, transform_from_pose_deg


@dataclass
class CaptureSet:
    """One calibration capture, in the units the maths wants them."""
    path: str
    n: int
    Q: np.ndarray                       # (l, n) joint angles, radians
    TCW_list: np.ndarray                # (l, 4, 4) camera -> artifact
    zeta_nominal: np.ndarray            # (4n,) [a | alpha | d | theta_off]
    T_0_6_controller: object            # (l, 4, 4) from the controller, or None
    w_rows: np.ndarray                  # (6l,) per-residual-row weights
    reproj_px: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _zeta_from_capture(data, n):
    """The DH table a capture recorded; lab_config's own only as a fallback."""
    current = C.build_zeta_nominal(n)
    if 'zeta_nominal' not in data:
        return current
    recorded = np.asarray(data['zeta_nominal'], dtype=float)
    if recorded.size != 4 * n:
        return current
    if not np.allclose(recorded, current):
        progress("NOTE: this capture recorded a different nominal DH table than "
                 "lab_config holds now; using the table the capture recorded")
    return recorded


def load_calibration_data(path=None):
    """Load a capture written by collect_data.py as a CaptureSet."""
    if path is None:
        path = C.data_path(C.CALIBRATION_DATA_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No capture at %s -- run collect_data.py first." % path)

    data = np.load(path, allow_pickle=True).item()
    if 'poses' not in data:
        raise ValueError(
            "%s predates the standardised capture format -- re-collect it." % path)

    n = int(data.get('n', C.N_DOF))
    poses = data['poses']
    if 0 < C.MAX_POSES < len(poses):
        progress("MAX_POSES = %d: fitting the first %d of %d poses"
                 % (C.MAX_POSES, C.MAX_POSES, len(poses)))
        poses = poses[:C.MAX_POSES]
    Q = joints_to_radians([p['robot_joints'][:n] for p in poses])
    TCW_list = np.array([p['T_C_W'] for p in poses], dtype=float)

    # A capture without the controller's own flange pose gives Stage F nothing
    # independent to be fitted against, so it is reported as absent, not guessed.
    if 'robot_pose_raw' in poses[0]:
        T_0_6_controller = np.array(
            [transform_from_pose_deg(p['robot_pose_raw']) for p in poses])
    else:
        T_0_6_controller = None

    return CaptureSet(
        path=path, n=n, Q=Q, TCW_list=TCW_list,
        zeta_nominal=_zeta_from_capture(data, n),
        T_0_6_controller=T_0_6_controller,
        w_rows=build_weight_vector(len(Q)),
        reproj_px=np.asarray([p.get('reproj_error_px', 0.0) for p in poses],
                             dtype=float))


def save_capture(path, poses, zeta_nominal, n=C.N_DOF, min_poses=None):
    """Write a capture in the one on-disk format -> ok; too-short ones are discarded."""
    poses = list(poses)
    floor = C.CAPTURE_MIN_POSES if min_poses is None else min_poses
    if len(poses) < floor:
        progress("DISCARDED: %d pose(s), fewer than the minimum of %d, so nothing "
                 "was written over %s" % (len(poses), floor, path))
        return False

    np.save(path, {'poses': poses,
                   'zeta_nominal': np.asarray(zeta_nominal, dtype=float),
                   'n': n}, allow_pickle=True)
    return True


def build_weight_vector(l):
    """Per-residual-row weights for l poses: [pos, pos, pos, ori x3] repeated."""
    return np.tile(
        [C.WEIGHT_POSITION] * 3 + [C.WEIGHT_ORIENTATION] * 3, l).astype(float)


def load_camera_intrinsics():
    """(K, dist) written by src/camera_calibration.py; raises if absent."""
    K_path = C.data_path(C.CAMERA_MATRIX_NAME)
    if not os.path.exists(K_path):
        raise FileNotFoundError(
            "No camera calibration at %s -- run src/camera_calibration.py first."
            % K_path)
    return (np.load(K_path),
            np.load(C.data_path(C.DIST_COEFFS_NAME)))


def load_validation_capture(path=None):
    """The hold-out set as a CaptureSet, or None if it was never captured.

    Same format as a calibration capture, so one loader reads both.
    """

    path = path or C.data_path(C.VALIDATION_DATA_NAME)
    return load_calibration_data(path) if os.path.exists(path) else None


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------

def load_results(path=None):
    """The results dict written by run_calibration.py."""
    path = path or C.data_path(C.RESULTS_NPY_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No calibration results at %s -- run run_calibration.py first." % path)
    return np.load(path, allow_pickle=True).item()


def _plain(value):
    """numpy scalars and arrays -> JSON-serialisable floats and lists."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def dh_table(zeta, n):
    """zeta as {name: value}, angles converted to degrees for human reading."""
    names = dh_param_names(n)
    out = {}
    for name, value in zip(names, np.asarray(zeta, dtype=float)):
        out[name] = float(np.rad2deg(value) if unit_of(name) == 'deg' else value)
    return out


def results_to_dict(results, n):
    """The results dict with numpy removed and DH tables given names and units."""
    plain = _plain({k: v for k, v in results.items()})
    plain['units'] = {'length': 'mm', 'angle': 'deg',
                      'hand_eye': '[x, y, z mm, roll, pitch, yaw deg]'}
    for key, zeta in [(k, v) for k, v in results.items()
                      if k.startswith('zeta_') and v is not None]:
        plain[key] = dh_table(zeta, n)
    return plain


def save_results_npy(path, results):
    """Store the full results dict, arrays intact."""
    np.save(path, results, allow_pickle=True)


def save_results_json(path, results, n):
    """Store the same results as readable JSON, DH parameters named."""
    with open(path, 'w') as f:
        json.dump(results_to_dict(results, n), f, indent=2)
        f.write('\n')


def save_text(path, text):
    """Write a report string to disk."""
    with open(path, 'w') as f:
        f.write(text)
