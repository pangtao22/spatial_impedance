# Spatial Impedance Simulations

Using Drake, this repo implements and compares two rotational stiffness definitions:
- [LinearBushingRollPitchYaw](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_linear_bushing_roll_pitch_yaw.html#Advanced_bushing_stiffness_and_damping),
- Axis-Angle/Quaternion with a 3x3 rotational stiffness matrix, as defined in chapter 3.3 of [Natale's book](https://www.springer.com/gp/book/9783540001591).