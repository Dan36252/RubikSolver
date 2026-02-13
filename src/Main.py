# This is the main program for the Rubik's Cube solving robot.
# This program will read the rubik's cube, and use a trained model and other functions to solve the cube.

import numpy as np
from CubeState import CubeState, MOVE_SEQUENCE
from ReadCube import ReadCube as ReadCube
from ModelRunner import Model
from torch import nn

cube_state = ReadCube() # NOT FINISHED

# Create a ModelRunner object
from ModelClass import RubikNN

model = Model()

def run_cube_data_test(states, moves):
    print("Running test...")
    num_correct_by_depth = np.zeros((150,))
    for sol_num in range(len(moves)):
        print(f"SOLUTION {sol_num}")
        for state_num in range(len(states[sol_num])):
            #print(f"State {state_num}")
            prev_moves = []
            for i in range(state_num):
                prev_moves.append(moves[sol_num][i][0])
            prediction = model.predict(np.array(states[sol_num][state_num]), prev_moves)
            actual = moves[sol_num][state_num][0]
            if prediction == actual:
                num_correct_by_depth[abs(int(moves[sol_num][state_num][1]))] += 1
            # if state_num == len(states[sol_num])-3:
            #     print(f"Prediction: {prediction}, Actual: {actual}")
            #     exit()
        if sol_num % 100 == 0:
            print(num_correct_by_depth)
    # for s in range(len(states)):
    #     print()

with open('RawData/training.seq.0', 'r') as file:
    print("Reading file...")
    all_solutions_states = []
    all_solutions_moves = []
    lsi = -1
    file_lines = file.readlines()
    states = []
    moves = []
    for l, line in enumerate(file_lines):
        if len(line) > 10:
            state = line.split()
            for i in range(len(state)):
                state[i] = int(state[i])
            states.append(state)
        else:
            move = [line.strip()]
            moves.append(move)
            if move[0] == "#":
                tsi = lsi + len(moves)
                num_moves = tsi-lsi
                for i in range(num_moves):
                    moves[i].append(str(num_moves-i))
                all_solutions_states.append(states)
                all_solutions_moves.append(moves)
                #print(f"Added solution #{len(all_solutions_states)}")
                states = []
                moves = []

    run_cube_data_test(all_solutions_states, all_solutions_moves)