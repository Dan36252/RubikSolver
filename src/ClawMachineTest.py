from ClawMachine import ClawMachine
from ReadCube import CubeReader
import time

robot = ClawMachine()
reader = CubeReader(robot)
reader.ReadCube()

# for i in range(7):
#     print("face to "+str(i))
#     robot.face_to_cam(i)
#     time.sleep(5)

# formula = ["L2", "B", "R2", "L'", "D'", "U'", "R'"]
#
# for m in formula:
#     robot.move(m)