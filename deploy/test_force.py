import logging
import time
from robot import UF850, ROBOT_IP

logging.basicConfig(level=logging.INFO)

robot = UF850(ip=ROBOT_IP, speed=30)
robot.connect()
arm = robot.arm

try:
    # Enable F/T sensor and zero it
    print("=== Enabling Force/Torque Sensor ===")
    arm.ft_sensor_enable(1)
    time.sleep(0.5)
    arm.ft_sensor_set_zero()
    time.sleep(0.3)
    print("Sensor zeroed. Place your hand on the end-effector to see force changes.\n")

    # Read force data in real-time
    print("=== Reading Force Data (press Ctrl+C to stop) ===")
    print(f"{'Fx':>8} {'Fy':>8} {'Fz':>8} {'Tx':>8} {'Ty':>8} {'Tz':>8}")
    print("-" * 54)

    while True:
        ft = arm.ft_ext_force  # [Fx, Fy, Fz, Tx, Ty, Tz]
        print(f"{ft[0]:8.2f} {ft[1]:8.2f} {ft[2]:8.2f} {ft[3]:8.2f} {ft[4]:8.2f} {ft[5]:8.2f}")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    arm.ft_sensor_enable(0)
    robot.disconnect()
