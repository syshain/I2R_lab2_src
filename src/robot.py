"""The only module that touches the xArm SDK: connect, read joints, read flange pose.

Opened with is_radian=True, so SDK angles are radians and lengths millimetres.
"""

import numpy as np

import lab_config as C


def connect(ip=None):
    """Connect, clear any error, put the arm in position mode -> the XArmAPI handle."""
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip or C.ROBOT_IP, is_radian=True, print_error=False)
    arm.set_state(state=0)
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
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
