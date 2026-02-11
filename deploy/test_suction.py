import logging
import time
from robot import UF850, ROBOT_IP

logging.basicConfig(level=logging.INFO)

robot = UF850(ip=ROBOT_IP, speed=30)
robot.connect()
arm = robot.arm

try:
    # 1. Read current GPIO state
    print("=== GPIO Diagnostics ===")
    ret, digitals = arm.get_tgpio_digital()
    print(f"Tool GPIO digital inputs:  ret={ret}, values={digitals}")
    ret, digitals = arm.get_tgpio_output_digital()
    print(f"Tool GPIO digital outputs: ret={ret}, values={digitals}")

    # 2. Test with set_vacuum_gripper (default hw_version=1: pin0=suction, pin1=release)
    print("\n=== Test 1: set_vacuum_gripper (hw_version=1) ===")
    arm.set_vacuum_gripper(True)
    time.sleep(30)
    ret, state = arm.get_vacuum_gripper()
    print(f"Vacuum state: ret={ret}, state={state}  (-1=off, 0=on no obj, 1=obj picked)")
    ret, digitals = arm.get_tgpio_output_digital()
    print(f"GPIO outputs after ON: {digitals}")
    arm.set_vacuum_gripper(False)
    time.sleep(1)

    # 3. Test with manual GPIO toggle (try reversed pins)
    print("\n=== Test 2: Manual GPIO - reversed pins ===")
    print("Setting pin0=0, pin1=1 (reversed)...")
    arm.set_tgpio_digital(0, 0)
    arm.set_tgpio_digital(1, 1)
    time.sleep(3)
    ret, digitals = arm.get_tgpio_output_digital()
    print(f"GPIO outputs: {digitals}")
    # Turn off
    arm.set_tgpio_digital(0, 0)
    arm.set_tgpio_digital(1, 0)
    time.sleep(1)

    # 4. Test with hardware_version=2 (pin3=suction, pin4=release)
    print("\n=== Test 3: set_vacuum_gripper (hw_version=2) ===")
    arm.set_vacuum_gripper(True, hardware_version=2)
    time.sleep(3)
    ret, state = arm.get_vacuum_gripper(hardware_version=2)
    print(f"Vacuum state (hw2): ret={ret}, state={state}")
    ret, digitals = arm.get_tgpio_output_digital()
    print(f"GPIO outputs after ON (hw2): {digitals}")
    arm.set_vacuum_gripper(False, hardware_version=2)
    time.sleep(1)

    print("\n=== Done ===")
    print("Which test activated the suction? Report back so we can fix robot.py")

finally:
    arm.set_vacuum_gripper(False)
    arm.set_vacuum_gripper(False, hardware_version=2)
    arm.set_tgpio_digital(0, 0)
    arm.set_tgpio_digital(1, 0)
    robot.disconnect()
