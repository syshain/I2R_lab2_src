"""Every tuneable for Lab 2. Lengths in millimetres, angles in degrees unless
the name says RAD. Frames: 0 = base, 6 = flange, C = camera, W = ArUco artifact.
"""

import os

import numpy as np


# =============================================================================
# HARDWARE  -- the only two values readable from the environment
# =============================================================================

ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.1.158")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

N_DOF = 6


# =============================================================================
# ARUCO ARTIFACT  -- geometry itself lives in data/aruco_config.json
# =============================================================================

ARUCO_DICT_NAME = "DICT_6X6_250"
ARUCO_MARKER_IDS = [0, 2, 3, 4, 5]   # checked against aruco_config.json at load

# Corner-order shift applied ONCE to each marker's model corners, as
# np.roll(corners, -roll). All zero matches every dataset collected so far.
CORNER_ROLL = {mid: 0 for mid in ARUCO_MARKER_IDS}

# Sub-pixel corner refinement before the pose is solved. Costs a few ms per
# frame and is what the existing datasets were captured with.
ARUCO_CORNER_REFINE_SUBPIX = True

# Views whose mean reprojection error exceeds this are refused at capture
# (collect_data.py); the looser figure is the live-preview flag in the detector.
COLLECT_REPROJ_REJECT_PX = 4.0
DETECT_REPROJ_REJECT_PX = 15.0

# Length of the axis arrows drawn on the live preview, mm (capture_view.py).
# They are the artifact frame W's own axes, drawn from its origin.
AXES_LENGTH_MM = 20.0

# The fewest poses collect_data.py will write: a pair may stop at 20 (Section
# 6.4), so a set of 20 or more is saved and a shorter one is refused.
CAPTURE_MIN_POSES = 20

# Run the calibration on only the first MAX_POSES of the saved set. 0 = all of
# them. Extension 3.1 sets this to 15; it overwrites data/calibration_results.*,
# which Task 4 then validates, so set it back to 0 afterwards.
MAX_POSES = 0

# =============================================================================
# CHESSBOARD  -- Task 2, camera_calibration.py; images go in data/calib_images
# =============================================================================
#
# INNER corners only. Measure SQUARE_SIZE_MM with a ruler before calibrating:
# a scale error there goes straight into fx and fy undetected.

CHESSBOARD_SIZE = (10, 7)
SQUARE_SIZE_MM = 15.0
CALIB_IMAGE_PATTERN = "calib_*.png"   # image_capture.py writes these into...
CALIB_IMAGE_FOLDER = "calib_images"   # ... data/calib_images, not data/ itself
CAM_CALIB_MIN_GOOD_IMAGES = 15        # handout section 6.3

# Handout section 6.3 thresholds, kept for reference; the judgement is the
# student's.
CAM_CALIB_RMS_GOOD_TARGET_PX = 0.5
CAM_CALIB_RMS_RECAPTURE_PX = 1.0


# =============================================================================
# NOMINAL DH PARAMETERS  (Modified / Craig convention)
# =============================================================================
#
# One row per joint i = 1..6, in the order of the joint-angle vector q:
#
#     [ theta_offset_deg,  d_mm,  alpha_deg,  a_mm ]
#
#   a_i     offset along x_{i-1} to the common normal of z_{i-1} and z_i
#   alpha_i rotation about x_{i-1} carrying z_{i-1} onto z_i
#   d_i     offset along z_{i-1} to where x_{i-1} crosses it
#   theta_i theta_offset_i + q_i, about z_{i-1}, carrying x_{i-1} onto x_i
#
# Craig convention (NOT classic Denavit-Hartenberg): the parameters attach to
# frame i-1 and the joint motion is the final theta term. Feeding these to a
# classic-DH routine pairs every a with the wrong alpha.

DH_NOMINAL_RAW = [
    [0.0,  243.3,   0.0,   0.0],    # joint 1
    [-90.0,   0.0, -90.0,   0.0],   # joint 2
    [-90.0,   0.0, 180.0, 200.0],   # joint 3
    [0.0,   227.6,  90.0,  87.0],   # joint 4
    [0.0,   0.0,    90.0,   0.0],   # joint 5
    [0.0,   61.5,  -90.0,   0.0],   # joint 6
]


def build_zeta_nominal(n=N_DOF):
    """The DH table above as zeta = [a | alpha | d | theta_off], mm and radians."""
    theta_off_deg, d_mm, alpha_deg, a_mm = [], [], [], []
    for row in DH_NOMINAL_RAW[:n]:
        theta_off_deg.append(row[0])
        d_mm.append(row[1])
        alpha_deg.append(row[2])
        a_mm.append(row[3])
    a = np.array(a_mm, dtype=float)
    alpha = np.deg2rad(np.array(alpha_deg, dtype=float))
    d = np.array(d_mm, dtype=float)
    theta_off = np.deg2rad(np.array(theta_off_deg, dtype=float))
    return np.concatenate([a, alpha, d, theta_off])


# DH parameters held at nominal during Stage 2. Names are dh_param_names()
# indices; see run_stages.py and observability.py.
LOCKED_TABLE_NAMES = ['a1', 'alpha1', 'd1', 'theta1',
                      'd6', 'd2', 'd3', 'theta6']


# =============================================================================
# TASK 1  --  NUMERICAL INVERSE KINEMATICS (ik_lite6.py)
# =============================================================================
#
# Everything in this section is in degrees and millimetres, the units the
# controller and UFactory Studio display. The IK residual is
# [position error in mm; orientation error in degrees] (handout section 5.2).

# The target flange pose: position mm, then roll, pitch and yaw in degrees in
# the convention UFactory Studio displays. Task 1(g) changes the position to
# UNREACHABLE_POSITION_MM and runs the guesses again.
TARGET_POSITION_MM = [-335.9, -91.5, 376.4]
TARGET_RPY_DEG = [-57.9, -40.5, 56.9]
TARGET_POSE = np.array(TARGET_POSITION_MM + TARGET_RPY_DEG, dtype=float)

UNREACHABLE_POSITION_MM = [-800.0, 0.0, 300.0]

# The six starting guesses of Table 1, degrees, spread over the workspace with
# E almost straight up and F with the elbow folded past itself. Add two of your
# own for the last two rows of Table 1.
START_GUESSES = [
    ('A', [190.0,  10.0,  60.0,   20.0, -60.0,   10.0]),
    ('B', [190.0, 100.0, 230.0,   90.0, -40.0,  -70.0]),
    ('C', [210.0,  30.0,  80.0, -140.0,  60.0, -150.0]),
    ('D', [190.0, 110.0, 240.0,  -60.0,  40.0,   90.0]),
    ('E', [180.0,   0.0,   4.0,     0.0, -45.0,   0.0]),
    ('F', [180.0,  90.0, 180.0,    0.0,   0.0,    0.0]),
]

# The limits set at every bench in the controller (Appendix B.1, second row),
# degrees. They are tighter than the manual's, and they are what the lab runs
# inside; J1 starts at 80 deg so the arm never swings through the cabling.
JOINT_LIMITS_DEG = np.array([[80.0, 270.0],
                             [-150.0, 150.0],
                             [-3.5, 300.0],
                             [-360.0, 360.0],
                             [-124.0, 124.0],
                             [-180.0, 180.0]])

# The joint readings the ik_lite6.py self-check builds its target from: the
# residual there must come out zero. It is guess A, which is reachable.
IK_CHECK_JOINTS_DEG = [190.0, 10.0, 60.0, 20.0, -60.0, 10.0]

# Finite-difference step for the IK Jacobian, degrees (about the same angle as
# FD_EPS is in radians). The calibration has its own step above.
IK_FD_EPS_DEG = 1e-5

# Two runs of ik_lite6.py count as one solution when their solved joint vectors
# agree to within this, in degrees on every joint. Different joint vectors that
# reach the same pose are different solutions: that is what Table 1 counts.
IK_SAME_SOLUTION_DEG = 0.1


# =============================================================================
# RESIDUAL WEIGHTS
# =============================================================================
#
# Per-row weights on the six pose residual rows (3 position mm, 3 orientation
# radians). Millimetres make the position rows ~10^2 larger than the radian
# ones, so these keep the two from competing on scale alone. They only bite
# where the two trade off, i.e. Stage 2.

WEIGHT_POSITION = 10.0
WEIGHT_ORIENTATION = 50.0

# Step size for the finite-difference Jacobian of the calibration (see
# IK_FD_EPS_DEG for the one the inverse kinematics uses).
FD_EPS = 1e-7


# =============================================================================
# OPTIMISER  -- defaults for lm_optimizer.levenberg_marquardt
# =============================================================================

LM_TOL = 1e-6         # stop when ||dx|| falls below this
LM_LAM0 = 1e-3        # initial damping
LM_MAX_ITER = 200     # cap when a caller does not pass one

# Damping schedule.
LM_LAM_FLOOR = 1e-12      # below this the step is effectively Gauss-Newton
LM_LAM_CEILING = 1e14     # above this the fit is declared stuck
LM_LAM_UP = 3.0           # multiplier after a rejected step
LM_LAM_DOWN_STRONG = 0.1  # divisor when a step cut the cost by > 10 %
LM_LAM_DOWN_MEDIUM = 0.3  # ... by > 1 %
LM_LAM_DOWN_WEAK = 0.5    # ... by less than 1 %
LM_STRONG_DROP = 0.1
LM_MEDIUM_DROP = 0.01

# Stopping detectors.
LM_STALL_REL = 1e-10   # relative cost drop below this counts as stalled
LM_CRAWL_LAM = 1e-8    # damping at or below this counts as sitting on the floor
LM_CRAWL_REL = 0.05    # ||dx|| changing by < 5 % ...
LM_CRAWL_COUNT = 3     # ... this many times running counts as converged


# =============================================================================
# PER-STAGE OPTIMISER SETTINGS
# =============================================================================

# Stage 1: hand-eye only, 6 parameters, against FK of the nominal DH table.
STAGE1_MAX_ITER = 200
STAGE1_REG_WEIGHT = 0.0    # well posed; no pull to the seed needed
STAGE1_SVD_THRESH = 0.0    # no observability mask

# Stage 2: partial-model DH + hand-eye, 22 parameters (16 free DH + 6 hand-eye).
# The mask is deliberately off: this Jacobian is finite differenced, so
# near-null directions are dominated by truncation error and masking on them
# stops the fit early. Stage F (hand-eye against the controller's own T_0_6) is
# the same problem as Stage 1 with a different reference frame, so it too
# reuses STAGE1_*.
STAGE2_MAX_ITER = 300
STAGE2_REG_WEIGHT = 1e-4   # weak L2 pull to nominal on 22 loose parameters
STAGE2_SVD_THRESH = 0.0
OBSERVABILITY_ZERO_REL = 1e-6


# =============================================================================
# DATA -- folder, filenames and path helpers; edit here, nowhere else
# =============================================================================

DATA_FOLDER = "data"

ARUCO_CONFIG_NAME = "aruco_config.json"
CAMERA_MATRIX_NAME = "camera_matrix.npy"
DIST_COEFFS_NAME = "dist_coeffs.npy"
CAMERA_RESULTS_NPZ_NAME = "camera_calibration.npz"    # everything at once
CAMERA_PARAMS_TXT_NAME = "camera_calibration.txt"     # ... and a readable copy
CAMERA_REPORT_FOLDER = "camera_report"                # corners drawn on each view

CALIBRATION_DATA_NAME = "calibration_data.npy"
RESULTS_NPY_NAME = "calibration_results.npy"
RESULTS_TXT_NAME = "calibration_results.txt"
RESULTS_JSON_NAME = "calibration_results.json"

VALIDATION_DATA_NAME = "validation_data.npy"
VALIDATION_NPY_NAME = "validation_results.npy"
VALIDATION_TXT_NAME = "validation_results.txt"

# Relocated-artifact hold-out set captured by validate_calibration.py.
VALIDATION_MIN_POSES = 8
VALIDATION_MAX_POSES = 12
VALIDATION_MIN_MOVE_MM = 50.0

_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_FOLDER)


def data_path(filename):
    """Absolute path to a file inside the data folder."""
    return os.path.join(DATA_DIR, filename)
