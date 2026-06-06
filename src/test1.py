import torch
import numpy as np
from NewModelRunner import Model, Solver
from CubeState import CubeState
import time

scramble = ["U"]#["R", "L", "F", "B", "R2", "D'", "U2", "L'", "B2"]
# scrambled = [3, 0, 2, 2, 0, 2, 4, 3, 4, 4, 0, 0, 5, 1, 3, 3, 1, 2, 0, 5, 1, 1, 2, 4, 5, 5, 1, 3, 4, 5, 5, 3, 3, 2, 2, 1, 0, 0, 5, 1, 4, 0, 0, 4, 5, 3, 2, 2, 1, 5, 3, 4, 4, 1]
# state = CubeState(f2l)
# for i in range(5):
#     for m in scramble:
#         state.move(m)

runner = Model()
# solver = Solver(scrambled)
# solver.solve_move_list()

# ============ Testing the Heuristic Model ================
with open("RawData/training.seq.1", "r") as file:
    for i, line in enumerate(file.readlines()):
        if len(line) > 40:
            state_list = []
            text_list = line.strip().split()
            for t in text_list: state_list.append(int(t))
            #print(state_list)
            state = CubeState(state_list)
            value = runner.get_state_value(state.encoded_data)
            print(value)
            if i > 300: break
        elif line.strip() == "#":
            print("===============================================")


# print("Thinking...")
# start = time.time()
# value = runner.get_state_value(state.flat_data)
# delta = time.time() - start
# print(f"MODEL PREDICTED VALUE: {value},   Time: {delta}")