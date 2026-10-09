"""ArUco artifact detection and monocular PnP: one camera frame -> T_C_W.

The board in data/aruco_config.json is what defines frame W.
"""

import json

import cv2
import numpy as np

import lab_config as C
from reporting import progress
from units import transform_from_rt

CORNER_COUNT = 4
MIN_POINTS_FOR_PNP = 4


class Detection:
    """One camera frame: whether the board was found, and what was found."""

    __slots__ = ('success', 'T_C_W', 'rvec', 'tvec', 'reproj_px', 'corners', 'ids')

    def __init__(self, success, T_C_W=None, rvec=None, tvec=None,
                 reproj_px=None, corners=None, ids=None):
        self.success = success
        self.T_C_W = T_C_W
        self.rvec = rvec
        self.tvec = tvec
        self.reproj_px = reproj_px
        self.corners = corners
        self.ids = ids


def load_artifact_config(path):
    """data/aruco_config.json -> (marker_ids, {id: 4 model corners in mm})."""
    with open(path) as f:
        cfg = json.load(f)
    artifact = next((t for t in cfg['toolList'] if t['id'] == 'ArucoBoard'),
                    cfg['toolList'][0])
    return (list(artifact['marker_ids']),
            {int(mid): np.array(raw, dtype=np.float64)
             for mid, raw in zip(artifact['marker_ids'],
                                 artifact['marker_corners_mm'])})


class ArucoArtifactDetector:
    """Detects the board in a camera frame and solves its pose T_C_W."""

    def __init__(self, camera_index=None, config_path=None, K=None, dist=None):
        self.camera_index = C.CAMERA_INDEX if camera_index is None else camera_index
        self.config_path = config_path or C.data_path(C.ARUCO_CONFIG_NAME)
        if K is None or dist is None:
            self.K = np.load(C.data_path(C.CAMERA_MATRIX_NAME))
            self.dist = np.load(C.data_path(C.DIST_COEFFS_NAME))
        else:
            self.K, self.dist = np.asarray(K, dtype=float), np.asarray(dist, dtype=float)

        self._build_model()

        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, C.ARUCO_DICT_NAME))
        parameters = cv2.aruco.DetectorParameters()
        if C.ARUCO_CORNER_REFINE_SUBPIX:
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("camera %d did not open" % self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_HEIGHT)

    def _build_model(self):
        """Model corners per marker: CORNER_ROLL applied once, then centred on W."""
        self.marker_ids, corners = load_artifact_config(self.config_path)
        self.marker_world = {
            mid: np.roll(raw, -C.CORNER_ROLL.get(mid, 0), axis=0)
            for mid, raw in corners.items()}
        # W is declared here: its origin is the centroid of every model corner and
        # its orientation the corner correspondence. Reprojection checks neither.
        centroid = np.vstack([self.marker_world[mid]
                              for mid in self.marker_ids]).mean(axis=0)
        for mid, raw in self.marker_world.items():
            self.marker_world[mid] = raw - centroid

        progress("artifact: %d markers, corner-set centroid taken as W's origin, "
                 "model from %s" % (len(corners), self.config_path))
        if sorted(self.marker_ids) != sorted(C.ARUCO_MARKER_IDS):
            progress("WARNING: board carries %s but lab_config.ARUCO_MARKER_IDS "
                     "says %s" % (sorted(self.marker_ids),
                                  sorted(C.ARUCO_MARKER_IDS)))

    def read(self):
        """One camera frame -> (ok, BGR frame)."""
        return self.cap.read()

    def close(self):
        """Release the camera."""
        self.cap.release()

    def _matched_points(self, corners, ids):
        """Pooled model and image points for every detected marker we know about."""
        object_points, image_points, matched = [], [], []
        for i, mid in enumerate(ids):
            model = self.marker_world.get(int(mid))
            if model is None:
                continue
            object_points.append(model)
            image_points.append(np.asarray(corners[i][0], dtype=np.float32))
            matched.append(int(mid))
        if not matched:
            return None, None, []
        return (np.vstack(object_points).astype(np.float32),
                np.vstack(image_points), matched)

    def detect(self, frame_bgr):
        """Board pose in the camera frame -> Detection with T_C_W (mm) and reproj_px.

        Any subset of the board's markers, from MIN_POINTS_FOR_PNP corners up.
        """

        if frame_bgr is None:
            return Detection(False)
        corners, ids, _ = self.detector.detectMarkers(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        if ids is None or len(ids) == 0:
            return Detection(False, corners=corners, ids=ids)

        object_points, image_points, matched = self._matched_points(
            corners, ids.flatten())
        if object_points is None or len(matched) * CORNER_COUNT < MIN_POINTS_FOR_PNP:
            return Detection(False, corners=corners, ids=ids)

        rvec, tvec = self._solve(object_points, image_points)
        if rvec is None:
            return Detection(False, corners=corners, ids=ids)

        projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.K, self.dist)
        reproj_px = float(np.mean(np.linalg.norm(
            projected.reshape(-1, 2) - image_points, axis=1)))
        if reproj_px > C.DETECT_REPROJ_REJECT_PX:
            return Detection(False, reproj_px=reproj_px, corners=corners,
                             ids=np.array(matched))

        rotation, _ = cv2.Rodrigues(rvec)
        return Detection(True, T_C_W=transform_from_rt(rotation, tvec.ravel()),
                         rvec=rvec, tvec=tvec.ravel(), reproj_px=reproj_px,
                         corners=corners, ids=np.array(matched))

    def _solve(self, object_points, image_points):
        """PnP on the pooled points -> (rvec, tvec), either of which is None."""
        for flags in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP):
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, self.K, self.dist, flags=flags)
            if ok:
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        object_points, image_points, self.K, self.dist, rvec, tvec)
                except cv2.error:
                    pass
                return np.asarray(rvec, dtype=float), np.asarray(tvec, dtype=float)
        return None, None
