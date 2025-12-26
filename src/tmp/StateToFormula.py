# This program is meant to take a Rubik's Cube state (i.e. configuration)
# and output a Formula that can solve it. Or, alternatively, it will
# output a single move, representing the best possible move that gets
# the cube closer to being solved.

import numpy as np
from CubeState import CubeState as CubeState
from ReadCube import ReadCube as ReadCube

dummy_state = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
])

cube_state = ReadCube()

cube1 = CubeState(dummy_state)
print(cube1)