import os
import numpy as np

import matplotlib.pyplot as plt
from pydrake.all import (MultibodyPlant, AddMultibodyPlantSceneGraph,
                         DiagramBuilder, LeafSystem, Parser,
                         ConnectMeshcatVisualizer, AngleAxis,
                         PortDataType, Quaternion, RotationMatrix, BasicVector,
                         Simulator, LogOutput, Adder,
                         SpatialForce, ExternallyAppliedSpatialForce)

from spatial_impedance_mappings import *


# %%
class SpatialImpedanceRegulator(LeafSystem):
    def __init__(self, Kp: np.ndarray, Kr: np.ndarray,
                 stiffness_mapping: SpatialStiffnessMappingInterface):
        """

        :param Kp: (3, 3) position stiffness
        :param Kr: (3, 3) rotational stiffness
        """
        super().__init__()
        self.set_name('spatial_impedance_controller')
        self.Kp = Kp
        self.Dp = 2 * 1.0 * np.sqrt(Kp * 1.0)
        self.Kr = Kr
        self.Dr = 2 * 1.0 * np.sqrt(Kr * 1.0)
        self.stiffness_mapping = stiffness_mapping

        self.state_input_port = self.DeclareInputPort(
            'state', PortDataType.kVectorValued, 13)
        self.damping_force_output_port = self.DeclareVectorOutputPort(
            'damping_force', BasicVector(6), self.calc_damping_force)
        self.stiffness_force_output_port = self.DeclareVectorOutputPort(
            'stiffness_force', BasicVector(6), self.calc_stiffness_force)

    def get_q_and_v_from_state(self, q_and_v):
        q = q_and_v[:7]
        v = q_and_v[7:]

        # W: world.
        # B: body.
        R_WB = RotationMatrix(Quaternion(q[:4] / np.linalg.norm(q[:4])))
        p_WB_W = q[4:]

        # angular velocity relative to world frame, expressed in world frame.
        w_WB_B = R_WB.multiply(v[:3])

        # translational velocity relative to world frame, expressed in world
        # frame.
        v_WB_W = v[3:]

        return R_WB, p_WB_W, w_WB_B, v_WB_W

    def calc_stiffness_force(self, context, output):
        q_and_v = self.state_input_port.Eval(context)
        R_WB, p_WB_W, w_WB_B, v_WB_W = self.get_q_and_v_from_state(q_and_v)

        spatial_force = self.stiffness_mapping(
            p_WB_W, R_WB, self.Kp, self.Kr)

        output.SetFromVector(spatial_force.get_coeffs())

    def calc_damping_force(self, context, output):
        q_and_v = self.state_input_port.Eval(context)
        R_WB, p_WB_W, w_WB_B, v_WB_W = self.get_q_and_v_from_state(q_and_v)

        spatial_force = np.zeros(6)

        # translational.
        spatial_force[:3] = - self.Dr @ w_WB_B

        # rotational.
        spatial_force[3:] = - self.Dp @ v_WB_W

        output.SetFromVector(spatial_force * 3)


# %%
builder = DiagramBuilder()

# plant
plant, sg = AddMultibodyPlantSceneGraph(builder, 0)
parser = Parser(plant, sg)
parser.AddModelFromFile(os.path.join(os.getcwd(), 'models', 'box_0.5m.sdf'))
plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
plant.Finalize()

# controller
controller = SpatialImpedanceRegulator(
    Kp=np.diag([100, 200, 400]), Kr=np.diag([10, 20, 40]),
    stiffness_mapping=calc_gimbal_stiffness_force)
adder = Adder(2, 6)
builder.AddSystem(controller)
builder.AddSystem(adder)
builder.Connect(plant.get_state_output_port(),
                controller.state_input_port)
builder.Connect(controller.stiffness_force_output_port,
                adder.get_input_port(0))
builder.Connect(controller.damping_force_output_port,
                adder.get_input_port(1))
builder.Connect(adder.get_output_port(),
                plant.get_applied_generalized_force_input_port())

# visualizer
body_idx = plant.GetBodyByName('box').index()
body_frame_id = plant.GetBodyFrameIdIfExists(body_idx)
viz = ConnectMeshcatVisualizer(builder, scene_graph=sg,
                               frames_to_draw=[body_frame_id],
                               axis_length=1.0, axis_radius=0.01)

# logging
state_log = LogOutput(plant.get_state_output_port(), builder)
state_log.set_publish_period(0.01)
diagram = builder.Build()

# simulation
sim = Simulator(diagram)
context = sim.get_mutable_context()
context_plant = plant.GetMyContextFromRoot(context)

# external force
ext_spatial_force = ExternallyAppliedSpatialForce()
ext_spatial_force.p_BoBq_B = np.zeros(3)
unit_tau = np.array([1, 0, 0], dtype=float)
unit_tau /= np.linalg.norm(unit_tau)
ext_spatial_force.F_Bq_W = SpatialForce(
    tau=33 * unit_tau,
    f=np.array([0, 0, 0], dtype=float))
ext_spatial_force.body_index = body_idx

plant.get_applied_spatial_force_input_port().FixValue(
    context_plant, [ext_spatial_force])

# simulation.
sim.set_target_realtime_rate(0.0)
sim.AdvanceTo(20.0)

#%%
n = len(state_log.sample_times())

rpy_log = np.zeros((n, 3))
angular_displacement = np.zeros(n)

for i in range(n):
    q = Quaternion(state_log.data()[:4, i])
    angle_axis = AngleAxis(q)
    rpy_log[i] = RollPitchYaw(q).vector()
    angular_displacement[i] = angle_axis.angle()


plt.figure()
t = state_log.sample_times()
plt.plot(t, rpy_log[:, 0], label='r')
plt.plot(t, rpy_log[:, 1], label='p')
plt.plot(t, rpy_log[:, 2], label='y')
plt.legend()
plt.show()

plt.figure()
plt.title('angular displacement')
plt.plot(t, angular_displacement)
plt.show()

#%%
n = 100
theta = np.linspace(0, np.pi, n)
omega_singular_values = np.zeros((n, 3))
r = np.array([1, 0, 0], dtype=float)

for i in range(n):
    Omega = 0.5 * np.cos(theta[i] / 2) * np.eye(3) \
                - 0.5 * np.sin(theta[i] / 2) * skew_symmetric(r)
    U, Sigma, Vh = np.linalg.svd(Omega)
    omega_singular_values[i] = Sigma


plt.figure()
plt.plot(theta, omega_singular_values[:, 0])
plt.plot(theta, omega_singular_values[:, 1])
plt.plot(theta, omega_singular_values[:, 2])
plt.show()
