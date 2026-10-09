"""Capture chessboard views for the camera calibration; SPACE saves, ESC quits.

Views land in data/calib_images as calib_NNN.png for camera_calibration.py.
"""

import os
import re

import cv2

import lab_config as C
from reporting import banner, field, progress

_VIEW_NAME = re.compile(r"calib_(\d+)\.png$")


def _lock_exposure_focus(cap):
    """Kill auto-exposure and autofocus: rolling AE jitters corners between frames."""
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)   # manual value, tune per camera


def next_index(out_dir):
    """Number for the next view: the highest calib_NNN.png on disk, gap-proof."""
    highest = 0
    for name in os.listdir(out_dir):
        match = _VIEW_NAME.match(name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


def main():
    """Live preview; SPACE writes the full frame as a PNG, ESC finishes."""
    out_dir = C.data_path(C.CALIB_IMAGE_FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    start = next_index(out_dir)

    banner("CHESSBOARD IMAGE CAPTURE")
    field("camera", "%d at %dx%d" % (C.CAMERA_INDEX, C.FRAME_WIDTH, C.FRAME_HEIGHT))
    field("output", out_dir)
    field("wanted", "%d views or more (handout section 6.3)"
          % C.CAM_CALIB_MIN_GOOD_IMAGES)
    if start:
        progress("%d views already on disk; this session adds from calib_%03d.png"
                 % (start, start + 1))
    progress("Hold the board 20-75 cm away, tilted every way it will go, and")
    progress("cover the four corners and the centre of the picture.")
    progress("SPACE saves the view, ESC finishes.")

    cap = cv2.VideoCapture(C.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_HEIGHT)
    _lock_exposure_focus(cap)

    count = start
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                progress("camera %d returned no frame" % C.CAMERA_INDEX)
                break
            view = cv2.resize(frame, (960, 540))
            cv2.putText(view, "views: %d" % count, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Chessboard capture: SPACE save, ESC quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                      # ESC
                break
            if key == 32:                      # SPACE
                count += 1
                name = "calib_%03d.png" % count
                cv2.imwrite(os.path.join(out_dir, name), frame)
                progress("saved %s" % name)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if count < C.CAM_CALIB_MIN_GOOD_IMAGES:
        progress("%d captured in total; the handout wants %d or more, so run"
                 % (count, C.CAM_CALIB_MIN_GOOD_IMAGES))
        progress("this again to add views -- earlier ones are kept.")
    progress("next: python src/camera_calibration.py")


if __name__ == "__main__":
    main()
