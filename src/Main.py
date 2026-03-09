from ClawMachine import ClawMachine
from ReadCube import CubeReader
from CubeState import CubeState
from DeepCubeA.search_methods.astar import bwas_python
from DeepCubeA.environments.cube3 import Cube3

print("Starting!")

# Initializing Claw Machine
robot = ClawMachine()

# Reading and reconstructing Cube
# reader = CubeReader(robot)
# face_colors = reader.ReadCube()
#
# # Flattening colors outputted by Cube Reader
# flat_colors = []
# for f in range(len(face_colors)):
#     for c in range(len(face_colors[f])):
#         flat_colors.append(shaped_data[f][c])
#
# print("Got reconstructed colors!")
# print(flat_colors)
# [1, 1, 0, 3, 1, 4, 5, 1, 3, 0, 1, 5, 5, 2, 5, 3, 3, 1, 5, 5, 2, 1, 4, 4, 2, 3, 2, 4, 2, 0, 2, 5, 5, 4, 3, 5, 4, 0, 0, 2, 3, 2, 3, 4, 3, 1, 0, 2, 0, 0, 0, 4, 4, 1]

# Temporary state generation, until Cube Reader is made more accurate
scramble = ["R", "U", "R'", "L", "D", "B2", "F'", "D'", "R2", "F", "B2", "L'"]
state = CubeState(encode=False)
for s in scramble:
    state.move(s)
flat_colors = state.flat_data

# Preparing data for DeepCubeA
cubestate = CubeState(data=flat_colors, encode=False)
deep_cube_state = cubestate.get_deepcube_data()

print("\n\nDeep Cube colors data:")
print(deep_cube_state)

print("\n\nSTARTING DEEPCUBE SOLUTION SEARCH!")
env = Cube3()
bwas_python(env, [deep_cube_state])
