from typing import Union

import magnum as mn
import numpy as np

from habitat_extensions.robots.base_robot import (
    Robot,
    RobotCameraParams,
    RobotParams,
)
from habitat_extensions.utils.geo_utils import (
    invert_transformation,
    transform_points,
)


class FetchRobot(Robot):
    def __init__(self, sim, params=None) -> None:
        if params is None:
            params = self.get_default_params()
        super().__init__(sim, params)

    @classmethod
    def get_default_params(cls) -> RobotParams:
        params = RobotParams(
            urdf_path="data/robots/hab_fetch/robots/hab_fetch.urdf",
            arm_joints=[15, 16, 17, 18, 19, 20, 21],
            gripper_joints=[23, 24],
            # arm_init_params=[-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005],
            arm_init_params=[-0.268, -1.133, 0.367, 1.081, 0.06, 2.002, 0.067],
            gripper_init_params=[0.0, 0.0],
            ee_link=22,
            ee_offset=mn.Vector3(0.08, 0.0, 0.0),
            # joint_stiffness=0.3,
            # joint_damping=0.3,
            # joint_max_impulse=10.0,
            arm_pos_gain=0.3,
            arm_vel_gain=0.3,
            arm_max_impulse=10.0,  # can not be too large when using pos gain
            cameras={
                "robot_arm_": RobotCameraParams(
                    cam_pos=mn.Vector3(0, 0, 0.1),
                    look_at=mn.Vector3(1.0, 0, 0.1),
                    up=mn.Vector3(0, 0, 1),  # +z is parallel to gripper
                    link_id=22,
                ),
                "robot_head_": RobotCameraParams(
                    cam_pos=mn.Vector3(0.17, 1.2, 0.0),
                    look_at=mn.Vector3(0.75, 1.0, 0.0),
                ),
                "robot_third_": RobotCameraParams(
                    cam_pos=mn.Vector3(-0.5, 1.7, -0.5),
                    look_at=mn.Vector3(1.0, 0.0, 0.75),
                ),
            },
        )

        # extra
        params.extra.update(
            torso_lift_joint=6,
            head_pan_joint=8,
            head_tilt_joint=9,
            hab2pyb=mn.Matrix4(
                np.array(
                    [
                        [1, 0, 0, -0.0036],
                        [0, 0.0107961, -0.9999417, 0],
                        [0, 0.9999417, 0.0107961, 0.0014],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            ),
            ee_com2link=mn.Matrix4.translation(
                mn.Vector3(-0.09, -0.0001, -0.0017)
            ),
        )

        return params

    @classmethod
    def get_params(cls, name) -> RobotParams:
        if name == "hab_fetch":
            return cls.get_default_params()
        else:
            raise KeyError(name)

    def reset(self):
        super().reset()

        # Initialize some joints following p-viz-plan
        self.set_joint_pos(self.params.extra["torso_lift_joint"], 0.15)
        self.set_joint_pos(self.params.extra["head_pan_joint"], 0.0)
        self.set_joint_pos(self.params.extra["head_tilt_joint"], np.pi / 2)

    def step(self):
        super().step()

        # Set some joint positions every step, following p-viz-plan
        self.set_joint_pos(self.params.extra["torso_lift_joint"], 0.15)
        self.set_joint_pos(self.params.extra["head_pan_joint"], 0.0)
        self.set_joint_pos(self.params.extra["head_tilt_joint"], np.pi / 2)

    def transform(self, x: np.ndarray, T: Union[str, np.ndarray]):
        """Transform point(s) (from habitat world frame) to specified frame."""
        if isinstance(T, str):
            if T == "world2base":
                T = np.array(self.base_T.inverted())
            elif T == "world2pbase":
                hab2pyb = self.params.extra["hab2pyb"]
                T = np.array(hab2pyb @ self.base_T.inverted())
            elif T == "base2pbase":
                hab2pyb = self.params.extra["hab2pyb"]
                T = np.array(hab2pyb)
            elif T == "base2world":
                T = np.array(self.base_T)
            elif T == "pbase2base":
                hab2pyb = self.params.extra["hab2pyb"]
                T = invert_transformation(hab2pyb)
            else:
                raise NotImplementedError(T)

        assert T.shape == (4, 4)
        return transform_points(x, T)

    def open_gripper(self):
        self.gripper_motor_pos = [0.04, 0.04]

    def close_gripper(self):
        self.gripper_motor_pos = [0.0, 0.0]
