## v2.4 - Ferdis fix list from Nicolos Lab 2 run
## 19 Nov 2026
- lm_optimizer: the stall test now fires only after an accepted step
    - Rejected first step no longer ends the solve at iteration 1 with calibration_results still written 
- aruco_artifact_detector: Detection carries only the matched markers' corners
    - Corners and ids always correspond and a foreign marker in view cannot crash drawDetectedMarkers
- capture_view: preview ids line uses np.asarray(...).ravel(); int() on the raw (N, 1) ids array fails under numpy 2

## v2.3 - Handout alignment fixes
## 12 Nov 2026
- Missing-camera-calibration guidance corrected: now points to image_capture.py + camera_calibration.py (handout Section 6.3, Part 1) instead of copying Lab 1 files (calibration_io, aruco_artifact_detector)
- ik_lite6.py ik_residual exercise now references handout Task 1(c) (was 1(b))
- STRUCTURE.md brought in line with the code: Lab_2_src layout, NotImplementedError stub description, five-model validation list, lab_config.data_path(), s_max grade bands removed

## v2.2 - Camera index for training lab
## 12 Nov 2026
- lab_config.py now has camera index default to 0 to work with the training lab computers

## v2.1 - Loader guards + ik_lite6 NotImplementedError stub
## 12 Nov 2026
- Camera matrix loads guarded with a "copy your Lab 1 files into Lab_2/data/" message (calibration_io.load_camera_intrinsics, aruco_artifact_detector)
- <TO_FILL> placeholders in ik_lite6.ik_residual replaced with a NotImplementedError stub; the module imports again

# v2.0 - Robot serial number reporting added
### 11 Nov 2026
- Add robot-interfacing programs now record the serial number for hardware uniqueness

# v1.4 - Minor fixes
### 13 Oct 2026
- Minor fixes - grades missing but not needed - removed 

# v1.3 - Dry run fixes
### 12 Oct 2026
- Changed ik_lite self-check to a more robust verification
- capture_view.py more robust to pc-dependencies
- lab_config.py now includes the updated joint angle limits
- observability.py now displays largest contributers to decomposed directions along with relative eigenvalues
    - Fixes dh parameters name sorting issue

# v1.2 - Requirements Corrections
### 11 Oct 2026
- Corrected requirements.txt to include precise module versions

# v1.1 - Corrections to v3, theory fixes
### 10 Oct 2026
- Removed set mode and state lines from robot.py
- camera calibration script crash fixed
- fk_lite6 now can be run standalone
- observability.py fixed
    - Rotation error was wrong
    - Position error was also mistaken. Fixed both
   
# v1 - First commit
### 9 Oct 2026
### Working lab 2 code 
- Inverse kinematics on Python
- Python based dh optimization 
    - Follows Sciavicco and Siciliano (2000)
- Python based hand eye solver compares against the above
- Python validation script for computed DH and hand eye
- Supporting scripts, modularized by function
- STRUCTURE.md in docs/ explains the codebase structure and instructions to run


