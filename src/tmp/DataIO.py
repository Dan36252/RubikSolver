import numpy as np
import re
from CubeState import MOVE_SEQUENCE
from pathlib import Path

def read_cubestates_file(filepath):
    X_train = None
    Y_train = None
    #solutionLine = "0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5"

    with open(filepath, 'r') as file:
        lines = file.readlines()
        states_list = []
        moves_list = []
        lsi = -1 # Last Solution Index in the moves_list, used for creating the "Distance till solution" value in label

        for line_i, line in enumerate(lines):
            # Add to states and moves lists. Include "solution lines" with the solved state and # move.

            if len(re.findall("[0-9]", line)) > 2:
                # Several digits found; line is a Cube State!
                # DON'T ignore solution lines; model needs to know how to stop!
                # Split string by whitespace, convert to list of ints
                state = []
                str_arr = line.split()
                for s in str_arr:
                    state.append(int(s))
                states_list.append(state)

            elif len(re.findall("[A-Z]", line)) > 0 or line.strip() == "#":
                # A letter was detected; line is a move!
                # Convert move like 'D2' to its corresponding int ID in MOVE_SEQUENCE
                move = []
                move.append(int(MOVE_SEQUENCE.index(line.strip())))
                moves_list.append(move)

                # Add the "Distance till solution" to labels
                if line.strip() == "#":
                    tmi = len(moves_list)-1  # This Move Index (in moves_list, the one that is '#')
                    num_moves = tmi - lsi # Total num of moves for this cube solution; including #
                    #print(f"Len moves_list: {len(moves_list)}")
                    #print(f"lsi: {lsi}, tmi: {tmi}, num_moves: {num_moves}")
                    for i in range(num_moves):
                        #print(f"adding to {lsi+i+1}")
                        moves_list[lsi+i+1].append(-int(num_moves-(i+1)))
                    lsi = tmi

        X_train = np.array(states_list)
        Y_train = np.array(moves_list)

    return X_train, Y_train

def read_multi_files():
    # X_train, Y_train = read_cubestates_file('training.seq.0')
    # write_cubestates_file('processed_data.txt', X_train, Y_train)
    X_train, Y_train = load_processed_data('processed_data.txt')
    print("Loaded file 0")
    X_train1, Y_train1 = read_cubestates_file('training.seq.1')
    print("Read file 1")
    write_cubestates_file('processed_data1.txt', X_train1, Y_train1)
    print("Wrote file 1")
    X_train = np.concatenate((X_train, X_train1))
    Y_train = np.concatenate((Y_train, Y_train1))
    X_train1, Y_train1 = read_cubestates_file('training.seq.2')
    print("Read file 2")
    write_cubestates_file('processed_data2.txt', X_train1, Y_train1)
    print("Wrote file 2")
    X_train = np.concatenate((X_train, X_train1))
    Y_train = np.concatenate((Y_train, Y_train1))
    X_train1, Y_train1 = read_cubestates_file('training.seq.3')
    print("Read file 3")
    write_cubestates_file('processed_data3.txt', X_train1, Y_train1)
    print("Wrote file 3")
    X_train = np.concatenate((X_train, X_train1))
    Y_train = np.concatenate((Y_train, Y_train1))

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

def load_processed_data(filepath):
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

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train

# def load_data():
#     raw_dir_path = Path('RawData')
#     processed_dir_path = Path('ProcessedData')
#
#     for raw_path in raw_dir_path.iterdir():
#         if raw_path.is_file():
#             file_num = re.search("[0-9]+", raw_path.name).group()
#             if file_num != None:
#                 #print(f"Yayy!! {file_num}")
#                 processed_path =
