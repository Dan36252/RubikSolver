# Importing Google Drive folder
# from google.colab import drive
# drive.mount('/content/drive')
# import sys
# sys.path.insert(0, '/content/drive/MyDrive/RubikSolver/src')
# sys.path.insert(0, '/content/drive/MyDrive/RubikSolver/src/RawData')
# sys.path.insert(0, '/content/drive/MyDrive/RubikSolver/src/ProcessedData')

import torch
import numpy as np
import re
from CubeState import MOVE_SEQUENCE
from pathlib import Path

def increase_tri_bit(single_bit):
    # Helper method for move_code_to_3_bits()
    single_bit += 1
    if single_bit > 1:
        single_bit = -1
        return single_bit, True # Carry-Over happened
    return single_bit, False # Carry-Over didn't happen

def move_code_to_3_bits(move_code):
    # Take a move code (1-19) and return its "3 tri-bit" representation.
    # In this format, '-1' is the new '0', and '0' and '1' are distinct higher values
    # Return as a list of 3 ints

    if move_code < 0 or move_code > 19:
        print("############  WARNING: ############")
        print(f"move_code_to_3_bits() found unexpected move_code: {move_code}. Should be [0-19]")

    # Create number
    l = [-1, -1, -1]
    for i in range(move_code):
        new_bit, carried = increase_tri_bit(l[2])
        l[2] = new_bit
        if carried:
            new_bit, carried = increase_tri_bit(l[1])
            l[1] = new_bit
            if carried:
                new_bit, carried = increase_tri_bit(l[0])
                l[0] = new_bit

    # Process number to switch -1 and 0 (nul state should be 0)
    for i in range(len(l)):
        if l[i] == -1:
            l[i] = 0
        elif l[i] == 0:
            l[i] = -1

    return l

def read_cubestates_file(filepath, device='cpu'):
    #solutionLine = "0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5"

    with open(filepath, 'r') as file:
        lines = file.readlines()
        states_list = []
        moves_list = []
        lsi = -1 # Last Solution Index in the moves_list, used for creating the "Distance till solution" value in label

        for line_i, line in enumerate(lines):
            # Add to states and moves lists. Include "solution lines" with the solved state and # move.
            # Also, fill the cube state lists with all the previous moves that created that state.

            #print(f"Reading Line {line_i}")

            if len(line) > 25:
                # Several digits found; line is a Cube State!
                # DON'T ignore solution lines; model needs to know how to stop!
                # Split string by whitespace, convert to list of ints
                state = []
                line = line.strip()
                str_arr = line.split()
                for s in str_arr:
                    state.append(int(s))
                states_list.append(state)

            else: #elif len(re.findall("[A-Z]", line)) > 0 or line.strip() == "#":
                # A letter was detected; line is a move!
                # Convert move like 'D2' to its corresponding int ID in MOVE_SEQUENCE
                move = []
                move_one_hot = np.zeros((len(MOVE_SEQUENCE)), dtype=np.int8)
                move_one_hot[MOVE_SEQUENCE.index(line.strip())] = 1
                for b in move_one_hot:
                    move.append(b)
                moves_list.append(move)

                # Add the "Distance till solution" to labels
                if line.strip() == "#":
                    #print(f"LSI = {lsi} -----------------------------------------------------------------")
                    tmi = len(moves_list)-1  # This Move Index (in moves_list, the one that is '#')
                    num_moves = tmi - lsi # Total num of moves for this cube solution; including #
                    for i in range(num_moves):
                        # Add the # moves till solution to each move in the move_list up to this solution move
                        moves_list[lsi+i+1].append(-int(num_moves-(i+1)))
                        # Now, go through all the cubestates up to this solution state and add a move in the chain
                        # that resulted in that state (filling the 30 prev moves for the model input).
                        # Loops with index i and index j are working together to achieve this
                        for j in range(30):
                            prev_move_code = 0
                            #print(f"Prev Move Index: {lsi+j-i}")
                            if i < num_moves-2:
                                if (i-j-1) >= 0:
                                    # lsi+j-i is the index of the previous move in the chain of moves that
                                    # we're currently checking for this cube state.
                                    prev_move_code = np.array(moves_list[lsi+i-j]).argmax()
                                prev_move_tri = move_code_to_3_bits(prev_move_code)
                                # Insert the "3 bit representation" of this prev move right after cube state
                                states_list[lsi+i+1].append(prev_move_tri[0])
                                states_list[lsi+i+1].append(prev_move_tri[1])
                                states_list[lsi+i+1].append(prev_move_tri[2])
                            else:
                                for k in range(3):
                                    states_list[lsi + i + 1].append(0)

                    lsi = tmi

        X_train = np.array(states_list, dtype=np.int8)
        Y_train = np.array(moves_list, dtype=np.int8)

        print("Praise God!")

    return X_train, Y_train

def write_cubestates_file(filepath, cubestates, label_moves):
    with open(filepath, 'w') as file:
        for i in range(len(cubestates)):
            state = cubestates[i]
            label = label_moves[i]

            state_str = ""
            for c in state:
                state_str += str(c) + " "

            label_str = ""
            for c in label:
                label_str += str(c) + " "

            file.write(state_str+"\n")
            file.write(label_str + "\n")

def load_processed_data(filepath, device='cpu'):
    X_train = []
    Y_train = []

    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if len(line) > 20:
                # Line is a cubestate
                color_list = line.split()
                num_list = []
                for c in color_list:
                    num_list.append(int(c))
                X_train.append(num_list)
            else:
                # Line is a move
                label_list = line.split()
                num_list = []
                for c in label_list:
                    num_list.append(int(c))
                Y_train.append(num_list)

    X_train = np.array(X_train, dtype=np.int8)
    Y_train = np.array(Y_train, dtype=np.int8)

    return X_train, Y_train

def load_data(device="cpu"):
    raw_dir_path = Path('RawData')
    processed_dir_path = Path('ProcessedData')
    # raw_dir_path = Path('/content/drive/MyDrive/RubikSolver/src/RawData')
    # processed_dir_path = Path('/content/drive/MyDrive/RubikSolver/src/ProcessedData')

    X_train = np.empty((0, 144), dtype=np.int8)
    Y_train = np.empty((0, 21), dtype=np.int8)

    # For each raw data file, check if processed file exists.
    # If it does, load it.
    # If it does not, read and write it.
    # Append to final X_train and Y_train with each step.
    counter = 0
    for raw_path in raw_dir_path.iterdir():
        if counter >= 10:
            break
        counter += 1
        if raw_path.is_file():
            # Get file number for this raw data file
            file_num_r = re.search("[0-9]+", str(raw_path)).group()
            if file_num_r is not None:
                # If file number exists, try to find processed file with same number
                loaded_processed = False
                for processed_path in processed_dir_path.iterdir():
                    # Get processed file number
                    file_num_p = re.search("[0-9]+", str(processed_path))
                    if file_num_p is not None:
                        # If processed number exists, check if matches raw file number
                        if file_num_r == file_num_p.group():
                            # Processed data match found! Load processed data
                            X_train1, Y_train1 = load_processed_data(str(processed_path), device)
                            # Append loaded data to final X_train and Y_train
                            X_train = np.concat((X_train, X_train1))
                            Y_train = np.concat((Y_train, Y_train1))
                            print(f"Loaded File {file_num_p} from processed data")
                            loaded_processed = True
                            break

                if not loaded_processed:
                    # If no corresponding processed file was found, read and write
                    X_train1, Y_train1 = read_cubestates_file(str(raw_path), device)
                    # Append read data to final X_train and Y_train
                    X_train = np.concat((X_train, X_train1))
                    Y_train = np.concat((Y_train, Y_train1))
                    print(f"Loaded File {file_num_r} from raw data")
                    # Write a processed file
                    write_cubestates_file((processed_dir_path/('processed_data'+str(file_num_r)+'.txt')), X_train1, Y_train1)

    return X_train, Y_train