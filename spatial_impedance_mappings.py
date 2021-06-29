from typing import Callable

import numpy as np

from pydrake.all import RotationMatrix, SpatialForce
from pydrake.math import RollPitchYaw

SpatialStiffnessMappingInterface = Callable[
    [np.ndarray, RotationMatrix, np.ndarray, np.ndarray], SpatialForce]


def skew_symmetric(w: np.ndarray):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=float)


def calc_axis_angle_stiffness_force(
        p_DB_D: np.ndarray, R_DB: RotationMatrix,
        Kp: np.ndarray, Kr: np.ndarray) -> SpatialForce:
    """
    Frame D: desired pose frame.
    Frame B: actual pose frame.
    :return:
    """
    # translational
    f = -Kp @ p_DB_D

    # rotational
    axis_angle = R_DB.inverse().ToAngleAxis()
    r = axis_angle.axis()
    theta = axis_angle.angle()
    o = np.sin(theta / 2) * r
    Omega = 0.5 * np.cos(theta / 2) * np.eye(3) \
            - 0.5 * np.sin(theta / 2) * skew_symmetric(r)
    tau = 4 * Omega.T @ Kr @ o
    # tau = Kr @ r * theta

    return SpatialForce(tau=tau, f=f)


def calc_gimbal_stiffness_force(
    p_DB_D: np.ndarray, R_DB: RotationMatrix,
    Kp: np.ndarray, Kr: np.ndarray) -> SpatialForce:

    rpy = RollPitchYaw(R_DB)
    r_angle, p_angle, y_angle = rpy.vector()

    # translational
    f = -Kp @ p_DB_D

    # rotational
    from numpy import sin, cos, tan
    N = np.array(
        [[cos(y_angle) / cos(p_angle), sin(y_angle) / cos(p_angle), 0],
         [-sin(y_angle), cos(y_angle), 0],
         [cos(y_angle) * tan(p_angle), sin(y_angle) * tan(p_angle), 1]])
    tau = -N.T @ Kr @ rpy.vector()

    return SpatialForce(tau=tau, f=f)


