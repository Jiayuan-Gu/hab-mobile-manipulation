import warnings
import numpy as np

try:
    import pybullet as p
except ImportError:
    warnings.warn("pybullet is not installed.")


def quat2mat(quat):
    """Convert quaternion to 3x3 matrix.
    The quaternion format is (x, y, z, w).
    """
    mat = p.getMatrixFromQuaternion(quat)
    return np.array(mat).reshape([3, 3])


def pose2mat(pos, quat):
    """Convert (position, quaternion) to 4x4 matrix.
    The quaternion format is (x, y, z, w).
    """
    T = np.eye(4)
    T[:3, :3] = quat2mat(quat)
    T[:3, 3] = pos
    return T


class PybulletRobot:
    """Pybullet wrapper over robot.

    Notes:
        If it is used to compute IK, please ensure that
        only joints intended to control are included in URDF.
    """

    def __init__(
        self, urdf_path, joint_indices, ee_link_idx, gui=False
    ) -> None:
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],  # urdf link position
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client_id,
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)

        # joint indices before end effector
        self.joint_indices = joint_indices
        # end effector link id
        self.ee_link_idx = ee_link_idx

        self.num_joints = p.getNumJoints(
            self.robot_id, physicsClientId=self.client_id
        )
        self.joint_limits = self.get_joint_limits()
        self.active_joints = self.get_active_joints()

    # ---------------------------------------------------------------------------- #
    # Interfaces
    # ---------------------------------------------------------------------------- #
    def get_base_pose(self):
        # center of mass
        return p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client_id,
        )

    def set_base_pose(self, position, orientation):
        # center of mass
        p.resetBasePositionAndOrientation(
            self.robot_id,
            position,
            orientation,
            physicsClientId=self.client_id,
        )

    def get_joint_states(self, joint_indices=None):
        if joint_indices is None:
            joint_indices = self.joint_indices
        return p.getJointStates(
            self.robot_id,
            joint_indices,
            physicsClientId=self.client_id,
        )

    def get_joint_positions(self, joint_indices=None):
        joint_states = self.get_joint_states(joint_indices)
        return [x[0] for x in joint_states]

    def set_joint_states(
        self, joint_positions, joint_velocities=None, joint_indices=None
    ):
        if joint_velocities is None:
            joint_velocities = np.zeros_like(joint_positions)
        if joint_indices is None:
            joint_indices = self.joint_indices
        assert len(joint_positions) == len(joint_indices)
        for i, joint_index in enumerate(joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_index,
                joint_positions[i],
                targetVelocity=joint_velocities[i],
                physicsClientId=self.client_id,
            )

    def get_link_state(self, link_index):
        return p.getLinkState(
            self.robot_id,
            link_index,
            computeForwardKinematics=1,
            physicsClientId=self.client_id,
        )

    @property
    def ee_state(self):
        return self.get_link_state(self.ee_link_idx)

    def IK(
        self,
        target_position,
        target_orientation=None,
        max_iters=20,
    ):
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_idx,
            target_position,
            targetOrientation=target_orientation,
            physicsClientId=self.client_id,
            maxNumIterations=max_iters,
        )
        return [joint_positions[i] for i in self.joint_indices]

    def compute_IK_error(self, target_ee_position, joint_positions):
        self.set_joint_states(joint_positions)
        return np.linalg.norm(self.ee_state[4] - np.array(target_ee_position))

    def clip_joint_positions(self, qpos):
        joint_indices = self.joint_indices
        assert len(qpos) == len(joint_indices)
        lower, upper = self.joint_limits
        return np.clip(qpos, lower[joint_indices], upper[joint_indices])

    # ---------------------------------------------------------------------------- #
    # Information
    # ---------------------------------------------------------------------------- #
    def get_joint_limits(self):
        lower = []
        upper = []
        for joint_index in range(self.num_joints):
            joint_info = p.getJointInfo(
                self.robot_id, joint_index, physicsClientId=self.client_id
            )
            if joint_info[8] > joint_info[9]:
                lower.append(-np.inf)
                upper.append(np.inf)
            else:
                lower.append(joint_info[8])
                upper.append(joint_info[9])
        return np.array(lower), np.array(upper)

    def get_active_joints(self):
        active_joints = []
        for joint_index in range(self.num_joints):
            joint_info = p.getJointInfo(
                self.robot_id, joint_index, physicsClientId=self.client_id
            )
            if joint_info[2] != 4:  # 4 is for fixed joint
                active_joints.append(joint_index)
        return active_joints

    def get_link_and_joint_names(self):
        link_names = []
        joint_names = []
        for joint_index in range(self.num_joints):
            joint_info = p.getJointInfo(
                self.robot_id, joint_index, physicsClientId=self.client_id
            )
            link_names.append(joint_info[12].decode())
            joint_names.append(joint_info[1].decode())
        return link_names, joint_names

    # ---------------------------------------------------------------------------- #
    # Debugging
    # ---------------------------------------------------------------------------- #
    def print_joint_info(self):
        for joint_index in range(self.num_joints):
            joint_info = p.getJointInfo(
                self.robot_id, joint_index, physicsClientId=self.client_id
            )
            print(joint_info)


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--show-info", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    urdf_path = (
        "habitat_extensions/assets/robots/hab_fetch/robots/hab_fetch_arm.urdf"
    )
    robot = PybulletRobot(urdf_path, [0, 1, 2, 3, 4, 5, 6], 10, gui=args.gui)

    if args.show_info:
        robot.print_joint_info()
        print("lower", robot.joint_limits[0][robot.joint_indices])
        print("upper", robot.joint_limits[1][robot.joint_indices])
        print("active joints", robot.active_joints)

        # ee_com2link
        ee_state = robot.ee_state
        print("ee_com2link", pose2mat(ee_state[2], ee_state[3]))

    # check initial states
    print("base_pose (init)", robot.get_base_pose())
    print("qpos (init)", robot.get_joint_positions())

    # set states
    # qpos0 = [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005]
    qpos0 = [-0.257, -1.33, 0.276, 1.279, 0.123, 2.081, 0.005]
    robot.set_joint_states(qpos0)
    qpos = robot.get_joint_positions()
    np.testing.assert_allclose(qpos, qpos0)
    ee_state = robot.ee_state
    print("ee_state", ee_state)

    # -------------------------------------------------------------------------- #
    # Test IK
    cur_pos = robot.ee_state[4]
    ik_qpos = robot.IK(cur_pos)
    print("IK qpos", ik_qpos)

    offset = np.array([0.0, 0.0, 0.1])
    tgt_pos = cur_pos + offset
    ik_qpos = robot.IK(tgt_pos, max_iters=50)
    robot.set_joint_states(ik_qpos)
    print("IK ee_state", robot.ee_state)
    np.testing.assert_allclose(robot.ee_state[4], tgt_pos, atol=1e-4)
    # -------------------------------------------------------------------------- #

    # print(robot.IK((0.5, 0.0, 1.0)))
    # print(robot.IK((0.5, 0.0, 1.0), (0, 0.707, 0, 0.707)))

    if args.gui:
        for i in range(10000):
            # p.stepSimulation()
            offset = np.array([0.0, 0.0, 0.015])
            tgt_pos = robot.ee_state[4] + offset
            ik_qpos = robot.IK(tgt_pos, max_iters=50)
            robot.set_joint_states(ik_qpos)
            # time.sleep(1.0 / 240.0)
            time.sleep(1.0 / 60)
        p.disconnect()


if __name__ == "__main__":
    main()
