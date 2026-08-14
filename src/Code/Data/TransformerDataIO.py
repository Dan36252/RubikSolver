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
def extract_from_raw_data(filepath, split_solutions=True, remove_jagged=True, truncate_mask=None, truncate_dir="after", include_truncate_pos=True, last_move_is_solution=True, chop_mask=None, include_chop_pos="first", encode_states=True, f2l_only=False, include_prev_move=True):
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
        longest_solution = 0
        for i in range(0, len(lines), 2):
            # Get current (state, move) pair
            state = lines[i].strip().split()
            state = [int(color) for color in state]
            move = MOVE_SEQUENCE.index(lines[i+1].strip())

            #state = CubeState(state).encoded_data if encode_states else state
            states.append(state)
            moves.append(move)

            # Group and process data by tracking solutions, if split_solutions = True or truncate_mask != None
            if track_solutions:
                # When the loop reaches the end of a solution...
                if move == SOLUTION_MOVE_CODE:
                    # Get index of the last (state, move) pair in this solution
                    s2 = int(i/2)

                    # Get the states and moves for this solution
                    solution_states = states[s1:s2+1]
                    solution_moves = moves[s1:s2+1]

                    # Truncate data
                    if not (truncate_mask is None):
                        for s in range(len(solution_states)):
                            if state_matches_mask(solution_states[s], truncate_mask):
                                inc = 1 if include_truncate_pos else 0
                                start_truncate = 0 if truncate_dir == "after" else s-inc+1
                                end_truncate = s+inc if truncate_dir == "after" else len(solution_states)+1
                                solution_states = solution_states[start_truncate:end_truncate]
                                solution_moves = solution_moves[start_truncate:end_truncate]
                                if last_move_is_solution: solution_moves[len(solution_moves)-1] = SOLUTION_MOVE_CODE
                                break

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

                    # Update longest solution length
                    if len(solution_states) > longest_solution: longest_solution = len(solution_states)

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
        if not (chop_mask is None):
            print("Secondary data chopping...")
            if split_solutions:
                for i in range(len(output_states)):
                    solution_states = output_states[i]
                    solution_moves = output_moves[i]
                    output_states[i], output_moves[i] = chop_data(solution_states, solution_moves)
            else:
                output_states, output_moves = chop_data(output_states, output_moves)

        # Recursive helper method for processing cube states
        def process_states(states_list, moves_list):
            if type(states_list[0][0]) != int and type(states_list[0][0]) != float:
                for i in range(len(states_list)):
                    process_states(states_list[i], moves_list[i])
            else:
                # Update states list:
                for i in range(len(states_list)):
                    # Get one-hot vector for prev move, or zero vector if i==0
                    prev_move_list = []
                    if include_prev_move:
                        prev_move_list = [0]*len(MOVE_SEQUENCE)
                        prev_move = 0 if i == 0 else moves_list[i-1]
                        prev_move_list[prev_move] = 1

                    # Encode cubestate if needed
                    state = states_list[i]
                    if encode_states or f2l_only:
                        cube_state = CubeState(data=states_list[i], encode=encode_states, f2l_only=f2l_only)
                        state = cube_state.encoded_data if encode_states else cube_state.flat_data

                    # Concat and replace state in states_list[i] with result
                    if include_prev_move: state = state + prev_move_list

                    # Update the states_list with the new cube state vector
                    states_list[i] = state

                # Update moves list:
                for i in range(len(moves_list)):
                    # Turn move codes into one-hot vectors
                    move = [0] * len(MOVE_SEQUENCE)
                    move[moves_list[i]] = 1
                    moves_list[i] = move

        # Processing of data: encoding, concatenating prev move, and filling jagged space with placeholder data
        if encode_states or f2l_only or include_prev_move:
            print("Data processing... (encoding, concatenating prev move)")
            process_states(output_states, output_moves)

        # Recursive helper method for removing jagged edges
        def remove_jagged(states_list, moves_list):
            if type(states_list[0][0]) != int and type(states_list[0][0]) != float:
                for i in range(len(states_list)):
                    remove_jagged(states_list[i], moves_list[i])
            else:
                lacking = longest_solution - len(states_list)
                states_placeholder = [0]*len(states_list[0])
                moves_placeholder = [0]*len(moves_list[0]) if type(moves_list[0]) == list else 0
                for i in range(lacking):
                    states_list.append(states_placeholder)
                    moves_list.append(moves_placeholder)

        # Removal of jagged array ends
        if remove_jagged and split_solutions:
            print("Removing jagged array ends...")
            remove_jagged(output_states, output_moves)

        # Finally, convert result into numpy array if possible
        if remove_jagged or (not split_solutions and chop_mask is None):
            print("Converting to numpy arrays...")
            output_states = np.array(output_states, dtype=np.int8)
            output_moves = np.array(output_moves, dtype=np.int8)

        print("Done!!!")
        return output_states, output_moves


def load_f2l_solver_data():
    states, moves = extract_from_raw_data("Data/Solver/RawData/training.seq.1", split_solutions=True, remove_jagged=True, truncate_mask=F2L_MASK, truncate_dir="after", include_truncate_pos=True, last_move_is_solution=True, chop_mask=None, include_chop_pos="first", encode_states=True, f2l_only=True, include_prev_move=True)
    counter = 0
    for i in range(len(states[0])):
        counter += 1
        print(states[0][i])
        all_zeros = True
        for j in range(len(states[0][i])):
            if states[0][i][j] != 0: all_zeros = False
        if all_zeros: break
    print(counter)
    print("IMPLEMENT: f2l solver data")
    # Longest solution (ONLY UP TO F2L): 71
    # training.seq.0: 67
    # training.seq.1: 66
    # training.seq.2: 67
    # training.seq.3: 67
    # training.seq.4: 71
    # training.seq.5: 64
    # training.seq.6: 70
    # training.seq.7: 71
    # training.seq.8: 68
    # training.seq.9: 65
    # training.seq.99: 64
    # Longest solution (All moves):
    # training.seq.0: 81
    # training.seq.1: 82



def load_last_layer_solver_data():
    print("IMPLEMENT: last layer solver data")