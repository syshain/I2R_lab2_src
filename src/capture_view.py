"""Live-preview drawing and the capture loop shared by collect_data and validate.

Presentation only: frames come from the detector, SPACE accepts a pose, nothing stores.
"""

import cv2
import numpy as np

import lab_config as C


def project_point(P, rvec, tvec, K, dist):
    """One 3D point in the artifact frame -> (u, v) integer pixels."""
    points, _ = cv2.projectPoints(np.array([P], dtype=np.float64),
                                  rvec, tvec.reshape(-1, 1), K, dist)
    return tuple(int(v) for v in points[0].ravel())


def draw_axes(frame, detection, K, dist, axis_length_mm=None):
    """Draw the artifact frame's axes, or a green dot if they leave the image."""
    length = C.AXES_LENGTH_MM if axis_length_mm is None else axis_length_mm
    u, v = project_point(np.zeros(3), detection.rvec, detection.tvec, K, dist)
    if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
        cv2.drawFrameAxes(frame, K, dist, detection.rvec,
                          detection.tvec.reshape(-1, 1), length)
    else:
        cv2.circle(frame, (u, v), 6, (0, 255, 0), -1)


def draw_markers(frame, detection):
    """Outline every detected marker and label it with its id."""
    if detection.corners is not None and len(detection.corners):
        cv2.aruco.drawDetectedMarkers(frame, detection.corners,
                                      detection.ids.reshape(-1, 1))


def capture_loop(detector, on_capture, max_poses=None, window="Capture",
                 hint="SPACE accept   ESC finish"):
    """Show the preview; return how many poses on_capture(detection) accepted.

    on_capture(detection) -> (accepted, note) and the note shows next frame.
    """

    accepted, note, note_ok = 0, "", True

    while True:
        ok, frame = detector.read()
        if not ok:
            raise RuntimeError("camera %d gave no frame" % detector.camera_index)

        detection = detector.detect(frame)
        if detection.success:
            draw_markers(frame, detection)
            draw_axes(frame, detection, detector.K, detector.dist)

        lines = []
        if not detection.success:
            lines.append(("artifact not found", (0, 0, 255)))
        if note:
            lines.append((note, (0, 255, 0) if note_ok else (0, 0, 255)))
        lines.append(("ids: %s   reproj: %s"
                      % ([int(i) for i in detection.ids]
                         if detection.ids is not None else [],
                         "%.2f px" % detection.reproj_px
                         if detection.reproj_px is not None else "-"),
                      (255, 255, 0)))
        lines.append(("poses: %d%s"
                      % (accepted, " / %d" % max_poses if max_poses else ""),
                      (255, 255, 0)))
        lines.append((hint, (255, 255, 0)))
        for row, (text, colour) in enumerate(lines):
            cv2.putText(frame, str(text), (10, 30 + 30 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord(' ') and (max_poses is None or accepted < max_poses):
            if not detection.success:
                note, note_ok = "artifact not found, nothing captured", False
            else:
                note, note_ok = on_capture(detection)
                if note_ok:
                    accepted += 1

    cv2.destroyWindow(window)
    return accepted
