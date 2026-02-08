import logging
import os
import time
import math
from configparser import ConfigParser
import numpy as np
from xarm.wrapper import XArmAPI

logger = logging.getLogger(__name__)

# Home joint angles in degrees (converted from simulation home_joints in radians)
HOME_JOINTS_DEG = [
    math.degrees(0.0),      # -90.0
    math.degrees(math.pi / 15),      # 12.0
    math.degrees(-2 * math.pi / 9),   # -40.0
    math.degrees(math.pi),            #  180.0
    math.degrees(2 * math.pi / 7),    #  51.4
    0.0,
]

CONF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robot.conf')
parser = ConfigParser()
parser.read(CONF_PATH)
ROBOT_IP = parser.get('xArm', 'ip')
class UF850:
    def __init__(self, ip=ROBOT_IP, speed=30):
        """
        Args:
            ip: Robot IP address.
            speed: Default joint speed in deg/s.
        """
        self.ip = ip
        self.speed = speed
        self.arm = None

    def connect(self):
        self.arm = XArmAPI(self.ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        logger.info(f"Connected to UF850 at {self.ip}")

    def disconnect(self):
        if self.arm:
            self.arm.disconnect()
            logger.info("Disconnected")

    def go_home(self, speed=None, wait=True):
        """Move to home joint configuration."""
        self.set_joints(HOME_JOINTS_DEG, speed=speed, wait=wait)
        logger.info("Moved to home position")

    def set_joints(self, angles_deg, speed=None, wait=True):
        """Move to target joint positions (degrees).

        Args:
            angles_deg: List of 6 joint angles in degrees.
            speed: Joint speed in deg/s. Uses default if None.
            wait: Block until motion completes.
        """
        speed = speed or self.speed
        ret = self.arm.set_servo_angle(
            angle=list(angles_deg),
            speed=speed,
            wait=wait,
        )
        if ret != 0:
            logger.warning(f"set_servo_angle returned {ret}")
        return ret

    def set_joints_rad(self, angles_rad, speed=None, wait=True):
        """Move to target joint positions (radians).

        Args:
            angles_rad: List of 6 joint angles in radians.
            speed: Joint speed in deg/s. Uses default if None.
            wait: Block until motion completes.
        """
        angles_deg = [math.degrees(a) for a in angles_rad]
        return self.set_joints(angles_deg, speed=speed, wait=wait)

    def get_joints(self):
        """Get current joint positions in degrees.

        Returns:
            np.ndarray of shape (6,) with joint angles in degrees.
        """
        ret, angles = self.arm.get_servo_angle()
        if ret != 0:
            logger.warning(f"get_servo_angle returned {ret}")
        return np.array(angles[:6])

    def get_joints_rad(self):
        """Get current joint positions in radians.

        Returns:
            np.ndarray of shape (6,) with joint angles in radians.
        """
        return np.deg2rad(self.get_joints())

    def get_position(self):
        """Get current end-effector position [x, y, z, roll, pitch, yaw].

        Returns:
            np.ndarray of shape (6,).
        """
        ret, pose = self.arm.get_position()
        if ret != 0:
            logger.warning(f"get_position returned {ret}")
        return np.array(pose[:6])

    def set_suction(self, on, wait=False, timeout=3):
        """Activate or deactivate the vacuum gripper.

        Args:
            on: True to activate, False to deactivate.
            wait: Wait until object is picked (only when on=True).
            timeout: Max wait time in seconds.
        """
        ret = self.arm.set_vacuum_gripper(on, wait=wait, timeout=timeout)
        status = "ACTIVATED" if on else "DEACTIVATED"
        logger.info(f"Suction {status}")
        print(f"Suction: {status}")
        return ret

    def get_suction_status(self):
        """Get the current vacuum gripper status.

        Returns:
            str: 'off', 'on (no object)', or 'on (object picked)'.
        """
        ret, state = self.arm.get_vacuum_gripper()
        status_map = {-1: "off", 0: "on (no object)", 1: "on (object picked)"}
        status = status_map.get(state, f"unknown ({state})")
        print(f"Suction status: {status}")
        return status

    def play_trajectory(self, trajectory_deg, speed=None, interval=0.0):
        """Execute a sequence of joint positions.

        Args:
            trajectory_deg: Array of shape (N, 6), each row is joint angles in degrees.
            speed: Joint speed in deg/s.
            interval: Delay in seconds between waypoints.
        """
        for i, joints in enumerate(trajectory_deg):
            logger.info(f"Waypoint {i + 1}/{len(trajectory_deg)}: {joints}")
            self.set_joints(joints, speed=speed, wait=True)
            if interval > 0:
                time.sleep(interval)

    def descend_until_contact(self, force_threshold=5, step_mm=0.25, speed=100):
        """Descend in Z until the force sensor exceeds the threshold.

        Args:
            force_threshold: Force in N to detect contact (positive value).
            step_mm: Step size in mm per iteration.
            speed: Cartesian speed in mm/s.
        Returns:
            bool: True if contact detected, False if error.
        """
        # Enable F/T sensor and zero it
        self.arm.ft_sensor_enable(1)
        time.sleep(0.5)
        self.arm.ft_sensor_set_zero()
        time.sleep(0.3)

        # Switch to servo mode for real-time cartesian control
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.1)

        mvpose = list(self.get_position())
        contact = False

        while self.arm.connected and self.arm.state != 4:
            fz = self.arm.ft_ext_force[2]
            print(f"Z={mvpose[2]:.1f}mm, Fz={fz:.2f}N")

            if fz <= -force_threshold:
                print(f"Contact detected! Fz={fz:.2f}N <= -{force_threshold}N")
                contact = True
                break

            mvpose[2] -= step_mm
            ret = self.arm.set_servo_cartesian(mvpose, speed=speed, mvacc=2000)
            if ret != 0:
                print(f"set_servo_cartesian error: {ret}")
                break
            time.sleep(0.01)

        # Back to position mode
        self.arm.ft_sensor_enable(0)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.set_mode(0)
        self.arm.set_state(0)
        return contact

    def touch_object(self, approach_joints, force_threshold=5, speed=None):
        """Descend until contact at the specified approach joints.

        Args:
            approach_joints: Joint angles (deg) for close to object position.
            force_threshold: Force in N to detect contact.
            speed: Joint speed in deg/s.
        """
        speed = speed or self.speed

        # Move close to object
        print(f"Approaching object: {approach_joints}")
        self.set_joints(approach_joints, speed=speed, wait=True)

        # Descend until force contact
        print("Descending until contact...")
        contact = self.descend_until_contact(force_threshold=force_threshold)

        if not contact:
            print("No contact detected")
        else:
            print("Contact detected")

    def pick_object(self, hover_joints, approach_joints, place_joints=None,
                    lift_joints=None, force_threshold=5, speed=None):
        """Pick object and optionally place it at a destination.

        Args:
            hover_joints: Joint angles (deg) for hover position above object.
            approach_joints: Joint angles (deg) for close to object position.
            place_joints: Joint angles (deg) for place destination. If None, just pick and lift.
            lift_joints: Joint angles (deg) for lift after pick. Uses hover_joints if None.
            force_threshold: Force in N to detect contact.
            speed: Joint speed in deg/s.
        """
        speed = speed or self.speed
        lift_joints = lift_joints or hover_joints

        # 1. Move to hover position above the object
        print(f"Moving to hover position: {hover_joints}")
        self.set_joints(hover_joints, speed=speed, wait=True)

        # 2. Move close to object
        print(f"Approaching object: {approach_joints}")
        self.set_joints(approach_joints, speed=speed, wait=True)

        # 3. Descend until force contact
        print("Descending until contact...")
        contact = self.descend_until_contact(force_threshold=force_threshold)

        if not contact:
            print("No contact detected, aborting pick")
            return

        # 4. Activate suction
        self.set_suction(on=True)
        time.sleep(0.5)

        # 5. Lift up
        print("Lifting object...")
        self.set_joints(lift_joints, speed=speed, wait=True)
        self.get_suction_status()

        # 6. Move to place position and release
        if place_joints is not None:
            print(f"Moving to place position: {place_joints}")
            self.set_joints(place_joints, speed=speed, wait=True)

            print("Releasing object...")
            self.set_suction(on=False)
            time.sleep(0.5)

            # 7. Retreat
            print("Retreating...")
            self.set_joints(lift_joints, speed=speed, wait=True)