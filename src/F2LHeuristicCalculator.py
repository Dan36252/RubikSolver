import numpy as np
from CubeState import CubeState

class Calculator:
    def __init__(self):
        self.f2l_mask = [0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1]

    def get_value(self, cubestate):
        value = 0
        print(cubestate.flat_data)
        for f in range(6):
            for c in range(9):
                i = (f*9)+c
                if self.f2l_mask[i] > 0 and cubestate.flat_data[i] == f:
                    value += 1
        return value

start_state = [1, 4, 5, 4, 0, 3, 1, 4, 2, 0, 0, 0, 2, 1, 5, 3, 0, 0, 4, 1, 1, 2, 2, 2, 1, 5, 3, 4, 0, 0, 1, 3, 3, 5, 4, 2, 5, 1, 3, 1, 4, 3, 5, 0, 4, 2, 2, 2, 5, 5, 3, 3, 5, 4]
state1 = CubeState(start_state)
state2 = CubeState()

calc = Calculator()

print(calc.get_value(state1))
print(calc.get_value(state2))