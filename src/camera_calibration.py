"""Fit the camera intrinsics from chessboard views; run after image_capture.py.

Reads data/calib_images/calib_*.png and writes camera_matrix.npy,
dist_coeffs.npy, the .npz/.txt record and annotated corner reports.
"""

import glob
import os

import cv2
import numpy as np

import lab_config as C
from reporting import banner, field, heading, progress


def find_boards(paths):
    """[(path, corners, (w, h))] for every view holding the full board."""
    found = []
    for path in paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            progress("unreadable, skipped: %s" % os.path.basename(path))
            continue
        ok, corners = cv2.findChessboardCorners(gray, C.CHESSBOARD_SIZE, None)
        if not ok:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        h, w = gray.shape
        found.append((path, corners, (w, h)))
    return found


def object_points():
    """The inner-corner grid as (w*h, 3) float32 in millimetres."""
    w, h = C.CHESSBOARD_SIZE
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    return objp * C.SQUARE_SIZE_MM


def calibrate(objp, image_points, image_size):
    """(K, dist, rms, rvecs, tvecs); CALIB_FIX_K3 stops k3 chasing noise."""
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], image_points, image_size, None, np.zeros((5, 1)),
        flags=cv2.CALIB_FIX_K3)
    return K, np.asarray(dist, dtype=float).ravel(), float(rms), rvecs, tvecs


def reprojection_errors(objp, image_points, rvecs, tvecs, K, dist):
    """Per view: (mean, max) pixel distance detected corner -> reprojected."""
    out = []
    for pts, rvec, tvec in zip(image_points, rvecs, tvecs):
        reproj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        err = np.linalg.norm(reproj.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
        out.append((err.mean(), err.max()))
    return np.array(out, dtype=float)


def rms_grade(rms):
    """Handout section 6.3 bands for the reprojection RMS, px."""
    if rms < C.CAM_CALIB_RMS_GOOD_TARGET_PX:
        return "EXCELLENT"
    if rms < C.CAM_CALIB_RMS_RECAPTURE_PX:
        return "GOOD"
    return "RETAKE - above %.1f px" % C.CAM_CALIB_RMS_RECAPTURE_PX


def write_reports(found, objp, rvecs, tvecs, K, dist):
    """Detected corners in green, reprojected in red, into data/camera_report."""
    out_dir = C.data_path(C.CAMERA_REPORT_FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    for (path, corners, _), rvec, tvec in zip(found, rvecs, tvecs):
        frame = cv2.imread(path)
        reproj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        cv2.drawChessboardCorners(frame, C.CHESSBOARD_SIZE, corners, True)
        for x, y in reproj.reshape(-1, 2):
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), frame)
    return out_dir


def save_all(K, dist, rms, found, per_view, image_size, n_read):
    """The five artefacts: two .npy, one .npz, one .txt, plus the report folder."""
    K_path = C.data_path(C.CAMERA_MATRIX_NAME)
    dist_path = C.data_path(C.DIST_COEFFS_NAME)
    np.save(K_path, K)
    np.save(dist_path, dist)
    np.savez(C.data_path(C.CAMERA_RESULTS_NPZ_NAME),
             K=K, dist=dist, reprojection_error_rms=rms,
             n_views_read=n_read, n_views_used=len(found),
             image_size=np.array(image_size),
             chessboard_size=np.array(C.CHESSBOARD_SIZE),
             square_size_mm=C.SQUARE_SIZE_MM,
             per_view_mean_px=per_view[:, 0], per_view_max_px=per_view[:, 1],
             paths=np.array([p for p, _, _ in found]))

    lines = ["CAMERA CALIBRATION (chessboard)", ""]
    lines += ["board           %d x %d inner corners, %.1f mm squares"
              % (C.CHESSBOARD_SIZE[0], C.CHESSBOARD_SIZE[1], C.SQUARE_SIZE_MM)]
    lines += ["views used      %d of %d read" % (len(found), n_read)]
    lines += ["image size      %d x %d px" % image_size]
    lines += ["reprojection    %.3f px RMS  (%s)" % (rms, rms_grade(rms))]
    lines += ["", "camera matrix K"]
    lines += ["  [%10.3f %10.3f %10.3f]" % tuple(row) for row in K]
    lines += ["", "distortion (k1 k2 p1 p2 k3)"]
    lines += ["  " + "  ".join("%10.6f" % v for v in dist)]
    lines += ["", "views fitted:"]
    for (path, _, _), (mean_px, _) in zip(found, per_view):
        lines += ["  %-24s mean %.3f px" % (os.path.basename(path), mean_px)]
    text = "\n".join(lines) + "\n"

    txt_path = C.data_path(C.CAMERA_PARAMS_TXT_NAME)
    with open(txt_path, 'w') as f:
        f.write(text)
    return [K_path, dist_path, C.data_path(C.CAMERA_RESULTS_NPZ_NAME), txt_path]


def main():
    """Locate boards in every saved view, fit K and distortion, write the results."""
    banner("CHESSBOARD CAMERA CALIBRATION")
    folder = C.data_path(C.CALIB_IMAGE_FOLDER)
    paths = sorted(glob.glob(os.path.join(folder, C.CALIB_IMAGE_PATTERN)))
    field("views", "%d files matching %s in %s"
          % (len(paths), C.CALIB_IMAGE_PATTERN, folder))
    if not paths:
        progress("none found -- run python src/image_capture.py first.")
        return

    found = find_boards(paths)
    progress("full board located in %d of %d views" % (len(found), len(paths)))
    if len(found) < C.CAM_CALIB_MIN_GOOD_IMAGES:
        progress("STOP: handout section 6.3 needs %d or more usable views and"
                 % C.CAM_CALIB_MIN_GOOD_IMAGES)
        progress("%d were found. Run image_capture.py again to add views --"
                 % len(found))
        progress("earlier ones are kept -- then re-run this script.")
        return

    sizes = [shape for _, _, shape in found]
    if len(set(sizes)) > 1:
        progress("views of differing sizes were read; keeping the majority size")
        target = max(set(sizes), key=sizes.count)
        found = [f for f in found if f[2] == target]

    objp = object_points()
    image_points = [corners for _, corners, _ in found]
    image_size = found[0][2]

    heading("FIT")
    K, dist, rms, rvecs, tvecs = calibrate(objp, image_points, image_size)
    per_view = reprojection_errors(objp, image_points, rvecs, tvecs, K, dist)

    field("fitted", "%d views" % len(found))
    field("RMS", "%.3f px" % rms, rms_grade(rms))
    worst = int(np.argmax(per_view[:, 0]))
    if per_view[worst, 0] > C.CAM_CALIB_RMS_RECAPTURE_PX:
        progress("worst view %s at %.2f px mean -- retake that one if you can"
                 % (os.path.basename(found[worst][0]), per_view[worst, 0]))
    print()
    for row in K:
        field("K", "[ %10.3f %10.3f %10.3f ]" % tuple(row))
    field("dist", " ".join("%.6f" % v for v in dist))

    report_dir = write_reports(found, objp, rvecs, tvecs, K, dist)
    for path in save_all(K, dist, rms, found, per_view, image_size, len(paths)):
        progress("wrote %s" % path)
    progress("wrote corner reports to %s" % report_dir)
    progress("next: python src/collect_data.py")


if __name__ == "__main__":
    main()
