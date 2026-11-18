"""
Action adapter for the existing MuJoCo Franka Panda scene.

Proposed intermediate action format for our custom π₀ dataset:
    action[0:7] = absolute Panda arm joint targets in radians
    action[7]   = normalized gripper openness, 0.0 = closed, 1.0 = open

This file only validates/converts commands. It does NOT run π₀ and the
self-test does NOT move or step the simulation. Before training, the same
action convention must be used consistently in every demonstration and in
the LeRobot π₀ dataset/configuration.
"""

from pathlib import Path

import mujoco
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "panda_pick_place.xml"
)

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR_NAME = "actuator8"
ACTION_DIM = 8


class PandaActionInterface:
    """Map a normalized 8-D action to the existing Panda actuators."""

    def __init__(self, model):
        self.model = model

        self.arm_actuator_ids = []
        self.arm_qpos_indices = []
        self.arm_ctrl_limits = []
        self.arm_joint_limits = []

        for joint_name, actuator_name in zip(
            ARM_JOINT_NAMES, ARM_ACTUATOR_NAMES
        ):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(
                    f"Missing expected Panda joint/actuator: "
                    f"{joint_name}/{actuator_name}"
                )

            self.arm_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
            self.arm_actuator_ids.append(int(actuator_id))

            if model.actuator_ctrllimited[actuator_id]:
                ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]
            else:
                ctrl_min, ctrl_max = model.jnt_range[joint_id]
            self.arm_ctrl_limits.append((float(ctrl_min), float(ctrl_max)))

            if model.jnt_limited[joint_id]:
                q_min, q_max = model.jnt_range[joint_id]
            else:
                q_min, q_max = -np.inf, np.inf
            self.arm_joint_limits.append((float(q_min), float(q_max)))

        self.arm_qpos_indices = np.asarray(self.arm_qpos_indices, dtype=int)
        self.arm_actuator_ids = np.asarray(self.arm_actuator_ids, dtype=int)
        self.arm_ctrl_limits = np.asarray(self.arm_ctrl_limits, dtype=np.float64)
        self.arm_joint_limits = np.asarray(self.arm_joint_limits, dtype=np.float64)

        self.gripper_actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
        )
        if self.gripper_actuator_id < 0:
            raise ValueError(
                f"Could not find gripper actuator '{GRIPPER_ACTUATOR_NAME}'."
            )

        if model.actuator_ctrllimited[self.gripper_actuator_id]:
            self.gripper_ctrl_min, self.gripper_ctrl_max = (
                model.actuator_ctrlrange[self.gripper_actuator_id]
            )
        else:
            raise ValueError(
                "Gripper actuator has no control limits; inspect the scene XML "
                "before allowing model-generated commands."
            )

    @staticmethod
    def validate_action(action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected action shape ({ACTION_DIM},), got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or infinity.")
        return action

    def action_to_ctrl(self, action):
        """
        Convert an 8-D action into (actuator_ids, control_values).

        Joint targets are clipped to actuator control limits. Gripper openness
        is clipped to [0, 1] and scaled to the XML actuator's control range.
        """
        action = self.validate_action(action)
        arm_targets = np.clip(
            action[:7],
            self.arm_ctrl_limits[:, 0],
            self.arm_ctrl_limits[:, 1],
        )
        openness = float(np.clip(action[7], 0.0, 1.0))
        gripper_ctrl = (
            self.gripper_ctrl_min
            + openness * (self.gripper_ctrl_max - self.gripper_ctrl_min)
        )

        actuator_ids = np.concatenate(
            [self.arm_actuator_ids, [self.gripper_actuator_id]]
        )
        controls = np.concatenate([arm_targets, [gripper_ctrl]])
        return actuator_ids, controls

    def apply_action(self, data, action):
        """
        Write one validated command to data.ctrl.

        Caller remains responsible for stepping MuJoCo, limiting how quickly
        targets change, checking contacts, and deciding when to stop.
        """
        actuator_ids, controls = self.action_to_ctrl(action)
        data.ctrl[actuator_ids] = controls
        return controls.copy()


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Scene XML not found: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    interface = PandaActionInterface(model)
    current_arm = data.qpos[interface.arm_qpos_indices].copy()

    # Test a command at the current arm pose with the gripper fully open.
    test_action = np.concatenate([current_arm, [1.0]])
    actuator_ids, controls = interface.action_to_ctrl(test_action)

    print("Panda action-interface self-test passed.")
    print("Action convention: [7 joint targets in radians, gripper openness 0..1]")
    print(f"Action dimension: {ACTION_DIM}")
    print(f"Arm actuator IDs: {interface.arm_actuator_ids.tolist()}")
    print(f"Gripper actuator ID: {interface.gripper_actuator_id}")
    print(f"Test action: {test_action}")
    print(f"Mapped actuator IDs: {actuator_ids.tolist()}")
    print(f"Mapped control values: {controls}")
    print(
        "\nNo simulation steps were run and the robot was not moved. "
        "This validates the adapter only; it is not yet π₀'s final dataset "
        "configuration or a trained policy."
    )


if __name__ == "__main__":
    main()
