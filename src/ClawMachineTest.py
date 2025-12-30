from ClawMachine import ClawMachine
import time

robot = ClawMachine()

for i in range(7):
    print("face to "+str(i))
    robot.face_to_cam(i)
    time.sleep(5)

# formula = ["L2", "B", "R2", "L'", "D'", "U'", "R'"]
#
# for m in formula:
#     robot.move(m)