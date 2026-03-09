from ClawMachine import ClawMachine
from ReadCube import CubeReader
from CubeState import CubeState
from DeepCubeA.search_methods.astar import bwas_python
from DeepCubeA.environments.cube3 import Cube3

print("Starting!")

robot = ClawMachine()

reader = CubeReader(robot)
cube_colors = reader.ReadCube()

print("Got reconstructed colors!")
print(cube_colors)

cubestate = CubeState(data=cube_colors, encode=False)
deep_cube_state = cubestate.get_deepcube_data()

print("\n\nDeep Cube colors data:")
print(deep_cube_state)

print("\n\nSTARTING DEEPCUBE SOLUTION SEARCH!")
env = Cube3()
bwas_python(env, [deep_cube_state])
