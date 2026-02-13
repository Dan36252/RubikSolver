import torch
import numpy as np
import re, time
from CubeState import MOVE_SEQUENCE
from pathlib import Path

WHITE_CROSS_MASK = "- - - - 0 - - 0 - - - - - - - - - - - - - - 2 - - 2 - - 3 - 3 3 3 - 3 - - - - - 4 - - 4 - - - - - 5 - - 5 -"
F2L_MASK = "- - - 0 0 0 0 0 0 - - - - - - - - - - - - 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 - - - 4 4 4 4 4 4 - - - 5 5 5 5 5 5"
SOLUTION_MASK = "0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5"

def state_matches_mask(state, mask):
    # State: string, flat data (len = 54)
    # mask: string, like WHITE_CROSS_MASK or F2L_MASK or SOLUTION_MASK above
    if type(state) == str:
        state = state.strip().split()
    mask = mask.strip().split()
    #print(state)
    #print(f"STATE LENGTH: {len(state)}, MASK LENGTH: {len(mask)}")
    for i in range(len(mask)):
        m = mask[i]
        if m == "-": continue
        s = state[i]
        if str(s) != str(m): return False
    # print("Matches Mask!")
    # print(state)
    return True

def read_cubestates_file(filepath, split_mask = "", skip_after_mask=False, device='cpu'):

    with open(filepath, 'r') as file:
        lines = file.readlines()
        states_list = []
        moves_list = []
        lsi = -1 # Last Solution Index in the moves_list, used for creating the "Distance till solution" value in label
        skip_to_next_solve = False
        skip_check = False
        skip_num = 0

        for line_i, line in enumerate(lines):
            # Add to states and moves lists. Include "solution lines" with the solved state and # move.
            # Also, fill the cube state lists with all the previous moves that created that state.

            #print(f"Reading Line {line_i}")

            if skip_to_next_solve and not skip_check:
                if len(line) < 25:
                    continue
                elif not state_matches_mask(line, SOLUTION_MASK):
                    # print("skipped")
                    continue
                else:
                    skip_check = True
                    skip_num += 1
                    # if skip_num % 100 == 0:
                    #     print(f"Solution Number {skip_num}")
                    #     print(states_list[len(states_list)-1])
                    #     print(moves_list[len(moves_list) - 1])
                    #     print(len(states_list), len(moves_list))
                    continue

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

                if skip_to_next_solve or skip_check:
                    skip_to_next_solve = False
                    skip_check = False

                # Add the "Distance till solution" to labels
                if line_i > 0 and state_matches_mask(states_list[len(states_list) - 1], split_mask):
                    # moves_list[len(moves_list) - 1] = "0"
                    # print(f"LSI = {lsi} -----------------------------------------------------------------")
                    tmi = len(states_list) - 1  # This Move Index (in moves_list, the one that is '#')
                    num_moves = tmi - lsi  # Total num of moves for this cube solution; including #
                    # print(f"NUMBER OF MOVES: {num_moves}")
                    # time.sleep(0.1)
                    for i in range(num_moves):
                        # Add the # moves till solution to each move in the move_list up to this solution move
                        moves_list.append(-int(num_moves - (i + 1)))

                    if skip_after_mask:
                        skip_to_next_solve = True

                    lsi = tmi

            else: #elif len(re.findall("[A-Z]", line)) > 0 or line.strip() == "#":
                #moves_list.append(line.strip())
                do_nothing = True # does nothing

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
            try:
                for c in label:
                    label_str += str(c) + " "
            except:
                label_str += str(label)

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
                Y_train.append(int(line.strip()))
                # label_list = line.split()
                # num_list = []
                # for c in label_list:
                #     num_list.append(int(c))
                # Y_train.append(num_list)

    print("File: "+filepath)

    X_train = np.array(X_train, dtype=np.int8)
    Y_train = np.array(Y_train, dtype=np.int8)

    return X_train, Y_train

def split_train_test(X_list, Y_list):
    # This method works if X_list and Y_list are lists OR numpy arrays :)
    X_train, Y_train, X_test, Y_test = [], [], [], []

    sample_every = 400
    index = 0
    while index < len(Y_list):
        if index % sample_every == 0:
            X_test.append(X_list[index])
            Y_test.append(Y_list[index])
        else:
            X_train.append(X_list[index])
            Y_train.append(Y_list[index])
        index += 1

    X_train = np.array(X_train, dtype=np.int8)
    Y_train = np.array(Y_train, dtype=np.int8)
    X_test = np.array(X_test, dtype=np.int8)
    Y_test = np.array(Y_test, dtype=np.int8)

    return X_train, Y_train, X_test, Y_test

def __load_data(device="cpu"):
    raw_dir_path = Path('RawData')
    processed_dir_path = Path('NewProcessedData')
    # raw_dir_path = Path('/content/drive/MyDrive/RubikSolver/src/RawData')
    # processed_dir_path = Path('/content/drive/MyDrive/RubikSolver/src/NewProcessedData')

    X_train = np.empty((0, 54), dtype=np.int8)
    Y_train = np.empty((0,), dtype=np.int8)
    X_test = np.empty((0, 54), dtype=np.int8)
    Y_test = np.empty((0,), dtype=np.int8)

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
                            loaded_X, loaded_Y = load_processed_data(str(processed_path), device)
                            X_train1, Y_train1, X_test1, Y_test1 = split_train_test(loaded_X, loaded_Y)
                            # Append loaded data to final X_train and Y_train
                            X_train = np.concat((X_train, X_train1))
                            Y_train = np.concat((Y_train, Y_train1))
                            X_test = np.concat((X_test, X_test1))
                            Y_test = np.concat((Y_test, Y_test1))
                            print(f"Loaded File {file_num_p} from processed data")
                            loaded_processed = True
                            break

                if not loaded_processed:
                    print(f"Could not find processed file {file_num_r}. Reading cubestate file.")
                    time.sleep(0.25)
                    # If no corresponding processed file was found, read and write
                    loaded_X, loaded_Y = read_cubestates_file(str(raw_path), F2L_MASK, True, device)
                    X_train1, Y_train1, X_test1, Y_test1 = split_train_test(loaded_X, loaded_Y)
                    # for i in range(500):
                    #     print(X_train1[i])
                    #     print(Y_train1[i])
                    # Append read data to final X_train and Y_train
                    X_train = np.concat((X_train, X_train1))
                    Y_train = np.concat((Y_train, Y_train1))
                    X_test = np.concat((X_test, X_test1))
                    Y_test = np.concat((Y_test, Y_test1))
                    print(f"Loaded File {file_num_r} from raw data")
                    # Write a processed file
                    write_cubestates_file((processed_dir_path/('processed_data'+str(file_num_r)+'.txt')), loaded_X, loaded_Y)

    return X_train, Y_train, X_test, Y_test


def load_data_f2l(device="cpu"):
    X_train, Y_train, X_test, Y_test = __load_data(device)
    print(f"X_train: {X_train.shape}")
    print(f"Y_train: {Y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"Y_test: {Y_test.shape}")
    return X_train, Y_train, X_test, Y_test

