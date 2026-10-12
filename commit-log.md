# v1.3 - Dry run fixes
## 12 Oct 2026
- Changed ik_lite self-check to a more robust verification
- capture_view.py more robust to pc-dependencies
- lab_config.py now includes the updated joint angle limits
- observability.py now displays largest contributers to decomposed directions along with relative eigenvalues
    - Fixes dh parameters name sorting issue

# v1.2 - Requirements Corrections
## 11 Oct 2026
- Corrected requirements.txt to include precise module versions

# v1.1 - Corrections to v3, theory fixes
## 10 Oct 2026
- Removed set mode and state lines from robot.py
- camera calibration script crash fixed
- fk_lite6 now can be run standalone
- observability.py fixed
    - Rotation error was wrong
    - Position error was also mistaken. Fixed both
   
# v1 - First commit
## 9 Oct 2026
### Working lab 2 code 
- Inverse kinematics on Python
- Python based dh optimization 
    - Follows Sciavicco and Siciliano (2000)
- Python based hand eye solver compares against the above
- Python validation script for computed DH and hand eye
- Supporting scripts, modularized by function
- STRUCTURE.md in docs/ explains the codebase structure and instructions to run


