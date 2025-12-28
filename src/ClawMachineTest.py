from ClawMachine import ClawMachine
import time

robot = ClawMachine()

robot.move("F2")

formula = ["L2", "B", "R2", "L'", "D'", "U'", "R'"]

for m in formula:
    robot.move(m)