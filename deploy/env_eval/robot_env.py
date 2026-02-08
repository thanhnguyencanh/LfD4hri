import math
import os
import time
import numpy as np
from gymnasium.spaces import Box
import gymnasium as gym
from config import *
from utils.utils import Utils
from utils.RealRobotUtils import UF850

# Home joint angles in radians
HOME_JOINTS_RAD = (0, np.pi / 15, -2 * np.pi / 9, np.pi, 2 * np.pi / 7, 0)


class ImitationLearning(gym.Env):
    """Real robot environment for policy deployment."""

    def __init__(self):
        self.dt = 1 / 400  # Control frequency reference
        self.home_joints = HOME_JOINTS_RAD
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

        # Target coordination for the task
        self.chosen_coordination = np.array([0.5512, 0.0555, -0.0071])
        self.object_orientation = np.array([0, 1, 0, 0])
        self.state = None
        self.action_type = None
        self.max_episode = None
        self.utils = Utils()
        self.curriculum_learning = None
        self.noise = True
        self.frame = 5
        self.episode = 0
        self.max_steps = None
        self.writer = None

        self.workspace_bound = np.array([
            BOUNDS[0][-1] - BOUNDS[0][0],
            BOUNDS[1][-1] - BOUNDS[1][0],
            BOUNDS[2][-1] - BOUNDS[2][0]
        ])

        # Initialize real robot
        self.real_robot = UF850()
        self.real_robot.connect()
        self._suction_state = False
        self._contact_detected = False

        # Read robot specs from config
        self._read_specs_from_config(os.path.abspath(
            os.path.join(self.utils.find_project_root("DRL"), '../asset/params/uf850_suction.xml')))

    def seed(self, seed):
        print('set seed')
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Reset environment and return initial observation."""
        print(f"Action Type: {self.action_type}")
        self._init_history()
        self.observation = None
        self.success_buffer = []
        self.sim_step = 0

        self.seed(self.episode)

        # Move robot to home position
        print("Moving robot to home position...")
        self.real_robot.go_home()
        self.real_robot.set_suction(on=False)
        self._suction_state = False
        self._contact_detected = False
        time.sleep(0.5)

        # Read initial state from real robot
        self._read_state()
        self.last_commanded_position = np.concatenate((self.state[:6], [0]))

        # Task state vector (using chosen_coordination as target)
        self.target = self.chosen_coordination + np.array([0, 0, 0])
        self.ee_obj = self._get_ee_coordination() - self.chosen_coordination
        self.obj_target = (self.chosen_coordination - np.array(
            [1, 1, self.chosen_coordination[-1]])) - self.target

        # Build task state vector (16D)
        self.task_state_vector = np.concatenate((
            self.ee_obj / self.workspace_bound,
            self.object_orientation,  # 4D quaternion
            self.obj_target / self.workspace_bound,
            self.chosen_coordination / self.workspace_bound,
            self.target / self.workspace_bound,
        ))

        # Build observation (37D)
        self.observation = np.concatenate((
            self.state,  # joint position (6) + joint velocity (6) = 12D
            np.zeros(2),  # suction state, contact state = 2D
            self.task_state_vector,  # 16D
            np.zeros(7),  # previous action = 7D
        ))

        self._save_data(np.zeros(7), "action")
        self._save_data(np.zeros(7), "action")
        self.episode += 1
        print("RESET complete")
        return self.observation

    def step(self, action):
        """Execute action on real robot."""
        print("----------")
        print("policy output: ", action)
        print("----------")

        self._read_state()
        self._save_data(self._get_ee_coordination(), "ee_history")
        self._save_data(self.object_orientation, "obj_history")
        self._save_data(action, "action")

        # Clip and convert velocity action to position command
        arm_joint_ctrl = np.clip(action, -1.0, 1.0)
        ctrl_feasible = self._velocity_to_position_control(arm_joint_ctrl)

        self._save_data(ctrl_feasible, "joint")

        # Extract joint angles and suction signal
        joint_angles_rad = ctrl_feasible[:-1]
        joint_angles_deg = [math.degrees(a) for a in joint_angles_rad]
        suction_signal = float(ctrl_feasible[-1]) > 0.04

        print("--------")
        print("joint angle (rad):  ", joint_angles_rad)
        print("joint angle (deg): ", joint_angles_deg)
        print("--------")

        # Send commands to real robot
        self.real_robot.set_joints(joint_angles_deg, wait=False)

        # Control suction
        if suction_signal != self._suction_state:
            self.real_robot.set_suction(on=suction_signal)
            self._suction_state = suction_signal

        # Small delay to allow robot to move
        time.sleep(0.02)

        # Read new state
        self._read_state()
        self.last_commanded_position = np.concatenate((self.state[:6], [ctrl_feasible[-1]]))

        # Check for contact using force sensor
        contact = self._check_contact()

        # Save contact history
        self._save_data([suction_signal, contact], "contact_history")
        self._save_data(self._get_ee_coordination(), "ee_history")
        self._save_data(np.concatenate([self.chosen_coordination, self.object_orientation,
                                        self.chosen_coordination]), "obj_history")

        # Update task state
        self.ee_obj = self._get_ee_coordination() - self.chosen_coordination
        self.obj_target = (self.chosen_coordination - np.array([1, 1, self.chosen_coordination[-1]])) - self.target

        # Compute reward (simplified for real robot - based on distance to target)
        ee_pos = self._get_ee_coordination()
        distance = np.linalg.norm(ee_pos - self.chosen_coordination)
        reward = -distance  # Negative distance as reward

        # Check termination
        done = False
        success = False
        if self.action_type == 0:  # Touch action
            if contact:
                done = True
                success = True
                print("Contact detected! Task complete.")

        # Check max steps
        if len(self.history["action"]) >= self.max_steps:
            done = True

        # Build task state vector
        self.task_state_vector = np.concatenate((
            self.ee_obj / self.workspace_bound,
            self.object_orientation,
            self.obj_target / self.workspace_bound,
            self.chosen_coordination / self.workspace_bound,
            self.target / self.workspace_bound,
        ))

        # Build observation
        self.observation = np.concatenate((
            self.state,
            [float(suction_signal), float(contact)],
            self.task_state_vector,
            self.history["action"][-1],
        ))

        self.success_buffer.append(success)
        info = {'log': self.success_buffer}

        return self.observation, reward, done, info

    def _read_state(self):
        """Read joint states from real robot."""
        # Get joint positions in radians
        joint_positions = self.real_robot.get_joints_rad()

        # Estimate velocities (simple difference - could be improved)
        if hasattr(self, '_prev_joint_positions') and self._prev_joint_positions is not None:
            joint_velocities = (joint_positions - self._prev_joint_positions) / self.dt
        else:
            joint_velocities = np.zeros(6)

        self._prev_joint_positions = joint_positions.copy()

        # Add small noise for robustness (optional)
        if self.noise and hasattr(self, 'np_random'):
            joint_positions = joint_positions + self.np_random.uniform(-0.005, 0.005, size=6)
            joint_velocities = joint_velocities + self.np_random.uniform(-0.05, 0.05, size=6)

        self.state = np.hstack((joint_positions, joint_velocities))

    def _get_ee_coordination(self):
        """Get end-effector position from real robot."""
        pos = self.real_robot.get_position()
        return pos[:3] / 1000.0  # Convert mm to meters

    def _check_contact(self):
        """Check for contact using force sensor."""
        try:
            # Enable force sensor if not already
            if not hasattr(self, '_ft_enabled') or not self._ft_enabled:
                self.real_robot.arm.ft_sensor_enable(1)
                time.sleep(0.1)
                self.real_robot.arm.ft_sensor_set_zero()
                time.sleep(0.1)
                self._ft_enabled = True

            # Read force in Z direction
            fz = self.real_robot.arm.ft_ext_force[2]
            contact = fz <= -3.0  # Contact threshold (adjust as needed)

            if contact and not self._contact_detected:
                print(f"Contact detected! Fz={fz:.2f}N")
                self._contact_detected = True

            return contact
        except Exception as e:
            print(f"Force sensor error: {e}")
            return False

    def _velocity_to_position_control(self, velocity_action: np.ndarray) -> np.ndarray:
        """Convert velocity action to position command."""
        clipped_velocity = np.clip(
            velocity_action, self.robot_vel_bound[:7, 0], self.robot_vel_bound[:7, 1]
        )
        target_position = self.last_commanded_position + clipped_velocity * 0.03

        safe_position_command = np.clip(
            target_position, self.robot_pos_bound[:7, 0], self.robot_pos_bound[:7, 1]
        )
        return safe_position_command

    def _save_data(self, data, attr: str):
        """Save data to history."""
        self.history[attr].append(data)

    def _init_history(self):
        """Initialize history buffers."""
        self.history = {
            "obj_history": [],
            "ee_history": [],
            "action": [],
            "contact_history": [],
            "target_obj": [],
            "joint": [],
        }

    def _read_specs_from_config(self, robot_configs: str):
        """Read robot joint limits from config file."""
        root, root_name = self.utils.get_config_root_node(config_file_name=robot_configs)
        robot_name = root_name[0]
        print(f"Robot: {robot_name}")

        self.robot_pos_bound = np.zeros([self.action_space.shape[0], 2], dtype=float)
        self.robot_vel_bound = np.zeros([self.action_space.shape[0], 2], dtype=float)
        self.robot_pos_noise_amp = np.zeros(self.action_space.shape[0], dtype=float)
        self.robot_vel_noise_amp = np.zeros(self.action_space.shape[0], dtype=float)

        for i in range(self.action_space.shape[0]):
            self.robot_pos_bound[i] = self.utils.read_config_from_node(
                root, "qpos" + str(i), "pos_bound", float
            )
            self.robot_vel_bound[i] = self.utils.read_config_from_node(
                root, "qpos" + str(i), "vel_bound", float
            )
            self.robot_pos_noise_amp[i] = self.utils.read_config_from_node(
                root, "qpos" + str(i), "pos_noise_amp", float
            )[0]
            self.robot_vel_noise_amp[i] = self.utils.read_config_from_node(
                root, "qpos" + str(i), "vel_noise_amp", float
            )[0]

    def close(self):
        """Clean up robot connection."""
        if hasattr(self, 'real_robot') and self.real_robot is not None:
            try:
                if hasattr(self, '_ft_enabled') and self._ft_enabled:
                    self.real_robot.arm.ft_sensor_enable(0)
                self.real_robot.set_suction(on=False)
                self.real_robot.go_home()
                self.real_robot.disconnect()
            except Exception as e:
                print(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
<<<<<<< HEAD
        self.close()
=======
        self.close()
>>>>>>> 3833404 (modified)
