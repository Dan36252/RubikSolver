from ClawMachine import ClawMachine
import time

robot = ClawMachine()

print("TEST 1")
robot.move("L")
time.sleep(3)
print("TEST 2")
robot.move("D'")
time.sleep(3)
print("TEST 3")
robot.move("U2")
time.sleep(3)
print("TEST 4")
robot.turn_cube("L")
time.sleep(3)
print("TEST 5")
robot.turn_cube("D")
time.sleep(3)