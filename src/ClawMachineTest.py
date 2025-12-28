from ClawMachine import ClawMachine
import time

robot = ClawMachine()
robot.move("F'")
formula = ["L2", "U", "D'", "U'", "R'"]

for m in formula:
    robot.move(m)