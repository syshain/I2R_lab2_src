"""The only module that touches the xArm SDK: connect, read joints, read flange pose.

Opened with is_radian=True, so SDK angles are radians and lengths millimetres.
"""

import numpy as np

import lab_config as C


def connect(ip=None):
    """Connect and clear any error -> the XArmAPI handle; the mode is left alone.

    Section 6.3 has the pair turn Manual Mode on in UFactory Studio and then
    run collect_data.py. motion_enable(True) or set_mode(0) here would drop the
    arm back into position mode, and the pair could no longer move it by hand.
    """
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip or C.ROBOT_IP, is_radian=True, print_error=False)
    arm.clean_error()
    return arm


def disconnect(arm):
    """Release the SDK handle."""
    arm.disconnect()


def joint_angles(arm, n=C.N_DOF):
    """Flange joint angles -> (ok, q radians); a short report is a fault, not angles."""
    code, q = arm.get_servo_angle()
    q = np.asarray(q, dtype=float)
    if code == 0 and q.size < n:
        code = -1
    return code == 0, q[:n]


def flange_pose(arm):
    """The controller's own T_0_6 -> (ok, [x, y, z mm, rx, ry, rz rad]).

    The SDK appends the joint angles after the pose, so only the first six count.
    """

    code, pose = arm.get_position()
    pose = np.asarray(pose, dtype=float)
    if code == 0 and pose.size < 6:
        code = -1
    return code == 0, pose[:6]
