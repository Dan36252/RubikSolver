from Code.Data.CubeState import MOVE_SEQUENCE, CubeState
import numpy as np

f2l_data_path = "Data/Solver/TransformerData/F2LSolver"
last_layer_data_path = "Data/Solver/TransformerData/LastLayerSolver"

WHITE_CROSS_MASK = "- - - - 0 - - 0 - - - - - - - - - - - - - - 2 - - 2 - - 3 - 3 3 3 - 3 - - - - - 4 - - 4 - - - - - 5 - - 5 -"
F2L_MASK = "- - - 0 0 0 0 0 0 - - - - - - - - - - - - 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 - - - 4 4 4 4 4 4 - - - 5 5 5 5 5 5"
SOLUTION_MASK = "0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5"

SOLUTION_MOVE_CODE = MOVE_SEQUENCE.index("#")

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

# This method reads a raw data file (in Data/Solver/RawData) and returns two ndarrays of custom processed data: states and moves.
def extract_from_raw_data(filepath, split_solutions=True, remove_jagged=True, truncate_mask=None, truncate_dir="after", include_truncate_pos=True, chop_mask=None, include_chop_pos="first", encode_states=True, f2l_only=False, include_prev_move=True):
    # filepath - Path to the raw data file to be read
    # split_solutions - If True, the output ndarray's first dimension will correspond to distinct cube solutions in the data.
    #                   If False, the output ndarray's first dimension will correspond to cube states from all solutions in the file.
    # truncate_mask - (If split_solutions=True) The cubestate mask where, when first detected in a solution, all data before or after this cubestate will not be included (within this solution).
    # truncate_dir - "after" or "before". Whether to drop data within a solution after or before the truncate_mask.
    # include_truncate_pos - Whether to include the cubestate that matched the truncate_mask in the output data.
    # chop_mask - If not None, an additional dimension will be added to the output data, which divides the data according to the chop_mask. Doesn't matter if split_solutions=True or False.
    # include_chop_pos - "first", "last", or "none". Whether to include the chop_mask position in the output data, and if so, where (in each chop block).
    # encode_states - If True, cubestates will be encoded by the CubeState class. If False, cubestates will be left as they are (like CubeState.flat_data).
    # include_prev_move - Whether to include a one-hot representation of the previous move that resulted in this cubestate when storing cubestates in the output ndarray. If True, this prev move will be concatenated with each cubestate vector.

    # Store processed output data in two lists
    output_states, output_moves = [], []

    # Open file
    with open(filepath) as file:
        # Temporary containers for holding all states and moves in this file
        states = []
        moves = []

        # Complete initial data slicing: solution split and truncate data. Loop over each (state, move) pair in this file.
        print("Initial reading, solution splitting, & truncating...")
        lines = file.readlines()
        num_lines = len(lines)
        progress_counter = 0
        s1, s2 = 0, 0  # pointers to track boundaries of solution sets
        track_solutions = split_solutions or not (truncate_mask is None)
        for i in range(0, len(lines), 2):
            # Get current (state, move) pair
            state = lines[i].strip().split()
            move = MOVE_SEQUENCE.index(lines[i+1].strip())

            #state = CubeState(state).encoded_data if encode_states else state
            states.append(state)
            moves.append(move)

            # Group and process data by tracking solutions, if split_solutions = True or truncate_mask != None
            if track_solutions:
                # When the loop reaches the end of a solution...
                if move == SOLUTION_MOVE_CODE:
                    # Get index of the last (state, move) pair in this solution
                    s2 = i/2

                    # Get the states and moves for this solution
                    solution_states = states[s1:s2+1]
                    solution_moves = moves[s1:s2+1]

                    # Truncate data
                    if not (truncate_mask is None):
                        for s in range(len(solution_states)):
                            if state_matches_mask(solution_states[s], truncate_mask):
                                inc = 1 if include_truncate_pos else 0
                                start_truncate = 0 if truncate_dir == "before" else s+inc+1
                                end_truncate = s-inc if truncate_dir == "before" else len(solution_states)+1
                                solution_states = solution_states[start_truncate:end_truncate]
                                solution_moves = solution_moves[start_truncate:end_truncate]

                    # Store the set of states and moves corresponding to this solution
                    if split_solutions:
                        output_states.append(solution_states)
                        output_moves.append(solution_moves)
                    else:
                        for s in range(len(solution_states)):
                            output_states.append(solution_states[s])
                            output_moves.append(solution_moves[s])

                    # Remember the index of the first (state, move) pair of the next solution
                    s1 = s2+1

            # Log progress
            progress_counter += 1
            if progress_counter >= 10000:
                progress_counter = 0
                print("Progress: %.3f/1.00" % (i/num_lines))

        # Helper method for the secondary data chopping
        def chop_data(states_list, moves_list):
            chopped_states, chopped_moves = [], []
            c1, c2 = 0, 0
            total_len = len(states_list)
            for i in range(total_len):
                if i == total_len:
                    c2 = total_len+1
                    chopped_states.append(states_list[c1:c2])
                    chopped_moves.append(moves_list[c1:c2])
                elif state_matches_mask(states_list[i], chop_mask):
                    if i == 0:
                        c1 = i if include_chop_pos == "first" else i+1
                    else:
                        c2 = i+1 if include_chop_pos == "last" else i
                        chopped_states.append(states_list[c1:c2])
                        chopped_moves.append(moves_list[c1:c2])
                        c1 = i+1 if not include_chop_pos == "first" else i
            return chopped_states, chopped_moves

        # Complete secondary data dicing: chop data. Further group data if chop_mask != None.
        print("Secondary data chopping...")
        if not (chop_mask is None):
            if split_solutions:
                for i in range(len(output_states)):
                    solution_states = output_states[i]
                    solution_moves = output_moves[i]
                    output_states[i], output_moves[i] = chop_data(solution_states, solution_moves)
            else:
                output_states, output_moves = chop_data(output_states, output_moves)

        # Helper method for processing cube states
        def process_states(states_list, moves_list):
            if type(states_list[0][0]) != int and type(states_list[0][0]) != float:
                for i in range(states_list):
                    process_states(states_list[i], moves_list[i])
            else:
                for i in range(states_list):
                    # get one-hot vector for prev move, or zero vector if i==0
                    # encode cubestate if needed
                    # concat and replace state in states_list[i] with result

        # Final processing of data: encoding, concatenating prev move, and filling jagged space with placeholder data
        print("Final data processing... (encoding, concatenating prev move, filling jagged space)")


        print(np.array(states).shape)
        print(np.array(moves).shape)


def load_f2l_solver_data():
    extract_from_raw_data("Data/Solver/RawData/training.seq.0.txt")
    print("IMPLEMENT: f2l solver data")

def load_last_layer_solver_data():
    print("IMPLEMENT: last layer solver data")