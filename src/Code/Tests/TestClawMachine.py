from ClawMachine import ClawMachine
from ReadCube import CubeReader
from CubeState import CubeState
import time

robot = ClawMachine()
reader = CubeReader(robot)
# start_state = [1, 4, 5, 4, 0, 3, 1, 4, 2, 0, 0, 0, 2, 1, 5, 3, 0, 0, 4, 1, 1, 2, 2, 2, 1, 5, 3, 4, 0, 0, 1, 3, 3, 5, 4, 2, 5, 1, 3, 1, 4, 3, 5, 0, 4, 2, 2, 2, 5, 5, 3, 3, 5, 4]
# start_cube = CubeState(start_state)
#reader.ScanManyFacesData(5, start_cube)

colors = reader.ReadCube()
print(colors)

# robot.claws["L"].extend()
# robot.claws["F"].extend()
# robot.claws["R"].extend()
# robot.claws["B"].extend()
# robot.claws["D"].extend()

# for i in range(7):
#     print("face to "+str(i))
#     robot.face_to_cam(i)
#     time.sleep(5)

# formula = ["L2", "B", "R2", "L'", "D'", "U'", "R'"]
#
# for m in formula:
#     robot.move(m)