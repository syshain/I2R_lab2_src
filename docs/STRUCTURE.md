# Lab 2 — Repository Structure

This file documents the canonical folder/module template used by this lab. 

Lab 2 is **inverse kinematics + kinematic calibration** in Python: Part A is a numerical IK solver for the Lite 6 (damped Levenberg–Marquardt against the same modified-DH FK you write in Lab 1 style), Part B extends Lab 1's eye-in-hand calibration into a **three-stage comparison** that jointly interrogates the hand-eye transform and the DH table against an ArUco artifact. All stages are scored by one consistency metric on captures. 

## Notation

All transforms use the worksheet's frame labels:

| Symbol | Meaning |
|--------|---------|
| `0`    | Robot base frame |
| `6`    | End-effector (flange) frame |
| `C`    | Wrist camera frame |
| `W`    | ArUco artifact frame |
| `board`| Chessboard (camera-calibration target) — distinct from the ArUco artifact |

Key transforms:
- `T_0_6` : base → end-effector (from FK on joint angles, or the controller's own `get_position` reading).
- `T_C_W` : camera → ArUco artifact (from PnP on detected markers).
- `T_6_C` : end-effector → camera (the hand-eye unknown).
- `T_0_W = T_0_6 @ T_6_C @ T_C_W` : reconstructed artifact position in the base frame; a correct calibration makes this constant across poses.

**Parameter vector.** The optimiser fits DH and hand-eye packed in one vector:

```
xi = [zeta | t6c] = [a1..a6, α1..α6, d1..d6, θ_off1..θ_off6, px, py, pz, ωx, ωy, ωz]
```

`zeta` is the modified (Craig) DH table laid out as `[a | alpha | d | theta_offset]` blocks; the last six entries pack `T_6_C` as `[position, rotation-vector]`. Millimetres and radians inside the maths, millimetres and degrees on disk (see Units).

**Formulation: still eye-in-hand, but fitted rather than solved.** The wrist camera rides the flange and watches a *static* artifact, so `T_0_W(i) = T_0_6(i) @ T_6_C @ T_C_W(i)` must repeat for every pose `i`. Lab 1 solved the AX = XB consequence of that directly with `cv2.calibrateHandEye`; Lab 2 fits the consistency cost itself with Levenberg–Marquardt (`lm_optimizer.py`, seeded by a two-pose hand-eye estimate), which is what lets DH errors enter the same fit.

**Stages** (`run_calibration.py` runs them all offline on one stored capture):

| Stage | Free parameters | FK model used | Notes |
|-------|-----------------|---------------|-------|
| Baseline | none — nominal DH, `T_6_C = I` | nominal DH | The score everything improves on |
| Stage 1 — hand-eye against nominal kinematics | `T_6_C` (6) | nominal DH FK | Same problem as Lab 1; well posed, no regularisation |
| Stage F — hand-eye against the controller FK | `T_6_C` (6) | the controller's own `T_0_6` | The controller is factory-calibrated, so this isolates the hand-eye term from DH error. Needs `robot_pose_raw` in the capture; scored both through the controller FK it was fitted in and through nominal DH |
| Stage 2 — partial-model DH + hand-eye (Sicilia) | 16 DH + 6 hand-eye (22) | DH being fitted | Eight parameters locked at nominal (`a1, α1, d1, θ1, d6, d2, d3, θ6` — unidentifiable once `T_6_C` is free), weak L2 pull toward nominal, seeded from Stage 1's `T_6_C` |

Every candidate is scored by `s_max` — the maximum over X/Y/Z of the per-axis standard deviation of the reconstructed `T_0_W` translation across poses — graded `EXCELLENT < 5 mm`, `GOOD < 10 mm`, `ACCEPTABLE < 20 mm`. `s_max` is self-consistency, not accuracy: any constant error in the chain is invisible to it, which is why validation happens on a *relocated* artifact.

## Top-level layout

```
Lab_2/
├── src/          # Active Python code. One flat directory of single-responsibility scripts.
├── data/         # Calibration artifacts (.npy/.npz/.json/.txt) + captured images. Ships with aruco_config.json and an empty calib_images/; everything else is produced at runtime.
├── docs/         # This structure guide.
└── requirements.text
└── commit-log.md # Human readable version history
```
Rules that make the template portable across labs:

1. **`src/` is the only place active Python code lives.** It is a *flat* directory — no nested packages, no `__init__.py`. Scripts may import shared helpers from sibling modules but exchange calibration results through files in `data/`. This keeps each stage independently runnable.
2. **`data/` holds binary calibration output, captured images and run results in human-readable format, not source.** Files here are produced by one stage and consumed by the next. Paths into `data/` are resolved relative to the repo via `_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'` in each script, never hardcoded absolute paths — so a script runs correctly no matter what the current working directory is.

## Module roles (Python pipeline)

The pipeline scripts run in order; each reads the previous stage's output from `data/` and writes its own.

| Script | Owns | Produces in `data/` |
|--------|------|---------------------|
| `image_capture.py` | Grabs chessboard views from the USB camera on demand (SPACE to capture, ESC to finish); kills auto-exposure and autofocus on open so rolling AE doesn't jitter corners between frames. | `calib_images/calib_NNN.png` |
| `camera_calibration.py` | Intrinsics + distortion from the chessboard views; writes K and dist each as `.npy`, an everything-at-once `.npz`, a readable `.txt`, and annotated per-view corner reports. | `camera_matrix.npy`, `dist_coeffs.npy`, `camera_calibration.npz`, `camera_calibration.txt`, `camera_report/` |
| `collect_data.py` | Live capture loop: the arm is hand-guided (manual mode), SPACE records one pose — joint angles, the controller's flange pose and `T_C_W` stored together; poses over the reprojection limit are refused. | `calibration_data.npy` |
| `run_calibration.py` | **Three-stage entry point**: baseline, our-FK-vs-controller model-disagreement table, Stage 1, Stage F, Stage 2, comparison table and each stage's % cut in Stage 1's `s_max`. No robot connection, so a capture re-runs freely. | `calibration_results.npy`, `calibration_results.json`, `calibration_results.txt` |
| `validate_calibration.py` | Live validation at a *relocated* artifact: captures 8–12 fresh poses, checks the artifact really moved, then scores every calibration in `calibration_results.npy` (Baseline / Stage 1 / Stage F / Stage 2) on poses none of them saw. Nothing is refitted. | `validation_data.npy`, `validation_results.npy`, `validation_results.txt` |
| `observability.py` | Task 3, standalone: SVD of the Jacobian at the point Stage 2 starts from — all 30 parameters free vs Sicilia's eight locked — printing every singular value divided by the largest, since the unidentifiable directions are small rather than zero. | *(none — console only)* |
| `ik_lite6.py` | Part A: numerical inverse kinematics (damped LM through an SVD) against `fk_lite6`, in the units the controller displays (mm, degrees). **The exercise file: it ships with two `<TO_FILL>` markers inside `ik_residual`, deliberately not valid Python — the script will not run until you have replaced both.** The residual counts one millimetre and one degree the same; that equality is the whole reason the weighting is what it is. | *(none — library module + runnable CLI)* |

Shared modules (imported, never run):

| Module | Owns |
|--------|------|
| `lab_config.py` | Single source of truth for every tunable: hardware (robot IP, camera index — the only two values readable from the environment), chessboard geometry, ArUco dictionary and marker IDs (cross-checked against `data/aruco_config.json` at load), the nominal DH table, Stage 2's locked-parameter names, per-stage optimiser settings (max iterations, L2 regularisation weight, SVD observability threshold), the `s_max` quality bands, every `data/` filename, and `data_path()`. |
| `units.py` | The only places degrees and radians cross over. Disk is mm + degrees (the Lab 1 convention); every calculation is mm + radians. |
| `so3.py` | SO(3)/SE(3) helpers: rotation log/exp maps (with the ~π branch handled), 4×4 ↔ 6-vector packing for `T_6_C`. |
| `fk_lite6.py` | Modified (Craig) D-H forward kinematics for the Lite 6: `modified_dh_transform`, `fk_from_zeta` (+ batch), the ζ layout, DH parameter names/units, and `locked_indices`. |
| `robot.py` | The only module that touches the xArm SDK (imported lazily, so non-robot scripts run without it). Connect, read joints, read flange pose; opened with `is_radian=True`. |
| `aruco_artifact_detector.py` | One camera frame → `T_C_W` by monocular PnP. The board in `data/aruco_config.json` is what defines frame W. |
| `capture_view.py` | Live-preview drawing and the SPACE capture loop shared by `collect_data.py` and `validate_calibration.py`. Presentation only — nothing stores. |
| `calibration_io.py` | Reading captures and writing results; nothing here computes anything. `CaptureSet` hands datasets over in the units the maths wants. |
| `calibration_model.py` | The chain's forward model: `T_0_W` for all poses from `xi`, residuals + Jacobian, and hand-eye seeding (two-pose; a from-flange variant seeds Stage F). |
| `lm_optimizer.py` | Generic Levenberg–Marquardt least squares: row weighting, L2 regularisation, optional SVD observability mask. Blind to what the parameters mean; read the sign contract in its header (`J = −dr/dξ`) before touching it. |
| `calibration_metrics.py` | Scoring: `T_0_W` consistency stats, quality grades, FK-vs-controller RMSE, the `T_6_C` difference between stages, and the artifact-displacement check for validation. |
| `reporting.py` | All console output and the text reports in one grammar: `..` process noise, `== TITLE ==` a stage, `-- Heading` a results block, `label   value` a number to read off. |
| `run_stages.py` | The three stage fits themselves — what each estimates against what, seeded from Stage 1. They estimate; they do not print or save. Driven by `run_calibration.py`. |

Stage F is skipped with a clear message when a capture was recorded without controller flange poses — re-collect with the current `collect_data.py` to fit against them.

## Units

Internal maths and all stored transforms use **millimetres + degrees on disk**, matching the Lab 1 capture format: joint angles and the controller's `[x, y, z, roll, pitch, yaw]` reading are stored in degrees, every translation in millimetres. Inside the code everything is radians: `calibration_io` converts the joints deg→rad once at load, `robot.py` opens the SDK with `is_radian=True`, and `collect_data.py` converts back to degrees for storage. `units.py` is the crossing point, and names in `lab_config.py` carry a `RAD` suffix when a value is not in the default units.

## How to run
Create a virtual environment in the main directory as follows
```bash
python3 -m venv .venv
```
Activate the virtual environment
```bash
source .venv/bin/activate
```
The command line should now have `(.venv)` preceding the prompt. If running the code for the first time, install all required Python libraries with the following command
```bash
pip install -r requirements.txt
```
Once the pip installer finishes, the code is ready to run.
Open `lab_config.py` and ensure the config parameters are correct. `ROBOT_IP` is found on the control box of the robot (it and `CAMERA_INDEX` can also be exported as environment variables on the bench).

Part A needs neither robot nor camera. Fill the two `<TO_FILL>` markers in `ik_lite6.py` first, then:
```bat
python.exe src\ik_lite6.py                                  # residual self-check
python.exe src\ik_lite6.py X Y Z R P Y                      # solve one pose (mm, degrees)
```

Run the Part B stages in order from the repository root (capture frames, estimate intrinsics, record pose pairs, fit all stages offline, then validate — the scripts resolve `data/` themselves, so no need to change directory):

```bat
python.exe src\image_capture.py
python.exe src\camera_calibration.py
python.exe src\collect_data.py
python.exe src\run_calibration.py
python.exe src\validate_calibration.py
```

`observability.py` answers the Task 3 identifiability question any time after `collect_data.py`.

Requires: Python 3.12+, NumPy ≥ 1.26, SciPy ≥ 1.11, OpenCV 4.13.0.92, xArm Python SDK 1.17.3, and a reachable UFACTORY Lite 6. The SDK is imported lazily inside `robot.py`, so everything except the live-robot scripts (`collect_data.py`, `validate_calibration.py`) runs without the arm connected.
