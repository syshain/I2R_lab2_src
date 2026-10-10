"""Capture hand-eye data: the arm is moved by hand, SPACE records one pose.

Each pose stores joints, the controller's flange pose and the solved artifact pose.
"""

import numpy as np

import lab_config as C
from aruco_artifact_detector import ArucoArtifactDetector
from calibration_io import load_camera_intrinsics, save_capture
from capture_view import capture_loop
from fk_lite6 import fk_from_zeta
from reporting import banner, field, progress
from robot import connect, disconnect, flange_pose, joint_angles


def record_pose(arm, detection, records, zeta_nominal, n=C.N_DOF):
    """Accept a frame -> (ok, note shown on the preview); poses stored in deg and mm."""
    if detection.reproj_px > C.COLLECT_REPROJ_REJECT_PX:
        return False, "refused: %.2f px > %.1f px" % (
            detection.reproj_px, C.COLLECT_REPROJ_REJECT_PX)

    ok_joints, q = joint_angles(arm, n)
    ok_pose, pose = flange_pose(arm)
    if not (ok_joints and ok_pose):
        return False, "refused: the arm gave no pose"

    records.append({
        'robot_joints': np.rad2deg(q),
        'robot_pose_raw': np.concatenate([pose[:3], np.rad2deg(pose[3:6])]),
        'T_0_6': fk_from_zeta(q, zeta_nominal, n),
        'T_C_W': detection.T_C_W,
        'reproj_error_px': detection.reproj_px,
        'marker_ids': np.asarray(detection.ids),
    })
    field("pose %d" % len(records),
          "flange (%7.1f, %7.1f, %7.1f) mm   reproj %.2f px   ids %s"
          % (*pose[:3], detection.reproj_px, [int(i) for i in detection.ids]))
    return True, "pose %d captured" % len(records)


def main(path=None):
    """Capture poses from camera and arm until ESC, then write them to path."""
    path = path or C.data_path(C.CALIBRATION_DATA_NAME)
    n = C.N_DOF
    zeta_nominal = C.build_zeta_nominal(n)

    K, dist = load_camera_intrinsics()
    detector = ArucoArtifactDetector(K=K, dist=dist)
    arm = connect()

    banner("Collect hand-eye data")
    field("arm", C.ROBOT_IP)
    field("camera", "%d (%d x %d)"
          % (detector.camera_index, C.FRAME_WIDTH, C.FRAME_HEIGHT))
    field("output", path)
    field("refuse above", "%.1f px reprojection" % C.COLLECT_REPROJ_REJECT_PX)
    progress("Move the arm so the artifact is clear, press SPACE once per pose, "
             "ESC to finish.")

    records = []

    def on_capture(detection):
        """One accepted frame -> (ok, note shown on the preview)."""
        return record_pose(arm, detection, records, zeta_nominal, n)

    try:
        capture_loop(detector, on_capture, window="Collect hand-eye data")
    except RuntimeError as exc:
        progress("capture ended early: %s" % exc)
        progress("saving the %d pose(s) captured so far" % len(records))
    finally:
        detector.close()
        disconnect(arm)

    saved = save_capture(path, records, zeta_nominal, n)
    banner("Capture %s" % ("saved" if saved else "discarded"))
    field("file", path)
    field("poses", len(records))
    if records:
        reproj = np.asarray([p['reproj_error_px'] for p in records])
        field("reproj mean", "%.2f px" % reproj.mean())
        field("reproj worst", "%.2f px" % reproj.max())


if __name__ == "__main__":
    main()
