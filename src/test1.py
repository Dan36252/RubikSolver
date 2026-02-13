import torch
import numpy as np
from NewModelRunner import Model
from CubeState import CubeState

scramble = ["U"]#["R", "L", "F", "B", "R2", "D'", "U2", "L'", "B2"]

state = CubeState()
for i in range(5):
    for m in scramble:
        state.move(m)

runner = Model()

value = runner.get_state_value(state.flat_data)

print(f"MODEL PREDICTED VALUE: {value}")

# LEFT OFF: Value network is kinda trash :/
# Data looks fine. Brainstorm different network architecture,
# or just use masking to get rough value (which isn't good because mid-formula states will have bad values).
# but even that is ok because we're looking several moves deep and getting best total values.
# which is rather resource expensive but whatever.