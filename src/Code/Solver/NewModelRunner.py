import torch
import torch.nn as nn
import numpy as np
from CubeState import CubeState, MOVE_SEQUENCE, SOLVED_STATE
from NewModelClass import F2LValueNN, EncodedValueNN, X_transform, device

def load_model(weights_path='EncodedF2LValueWeights.pth'):
    model = EncodedValueNN().to(device)
    model.load_state_dict(torch.load(weights_path, weights_only=True), strict=False)
    return model

class Model:
    def __init__(self):
        self.model = load_model()

    def get_state_value(self, encoded_data):
        # state_list is a single cube state of length 54, as a list.
        self.model.eval()
        # cubestate = CubeState(state_list)
        #print(encoded_data)
        x = torch.from_numpy(np.array(encoded_data, dtype=np.int8).astype(np.float32))
        x = X_transform(x)
        #print(x)
        x = x.to(device)
        pred = self.model.forward(x)
        return pred

    # def predict(self, state):
    #     # state = numpy list of shape (54,)
    #     # prev_moves = list of Move Letters (R', U2, etc.). NOT in reverse order. (1st move, 2nd, ...)
    #     state = transform_X(state)
    #     prev_moves = transform_moves_list(prev_moves)
    #     # print("Calculated Prev Moves:")
    #     # print(prev_moves)
    #     X = torch.cat((state, prev_moves))
    #     self.model.eval()
    #     logits = self.model.forward(X)
    #     pred = MOVE_SEQUENCE[logits[:-1].argmax()]
    #     # print("Prediction:")
    #     # print(pred)
    #     return pred

class Solver:
    def __init__(self, initial_state_list):
        self.initial_state = CubeState(initial_state_list)
        self.value_model = Model()

    def solve_move_list(self):
        # Main method. Runs solver and returns a list of moves to get to the solution.
        print("Solving...")
        explore_queue = []

        #self.queue_all_next_states(explore_queue, self.initial_state)
        self.queue_all_next_branches(explore_queue, self.initial_state)
        examining = explore_queue.pop(0)
        num_explorations = 0  # keep track of how long algo has been running
        while examining != SOLVED_STATE and num_explorations <= 10000:
            self.queue_all_next_branches(explore_queue, examining)
            explore_queue = explore_queue[:5000]
            num_explorations += 1
            examining = explore_queue.pop(0)
            print(f"{num_explorations} explorations, current cost: {examining.total_cost}, current depth: {examining.g_cost}, queue len: {len(explore_queue)}")
            if num_explorations % 20 == 0:
                print("PRINTING QUEUE COSTS")
                for i in range(min(len(explore_queue), 20)):
                    print(explore_queue[i].g_cost)

    def get_all_next_states(self, cubestate):
        new_states = []
        for m in MOVE_SEQUENCE:
            if m != "-" and m != "#":
                s = cubestate.spawn_move(m)
                # total_cost = self.get_total_cost(s)
                # insert_index = len(queue_list)
                # for i in range(len(queue_list)):
                #     this_cost = queue_list[i][0]
                #     if total_cost < this_cost:
                #         insert_index = i
                #         break
                new_states.append(s)
        return new_states

    def queue_all_next_branches(self, queue_list, cubestate):
        # 1. Fill a temp array with all terminal nodes of n-deep branches starting from the given initial node
        # 2. Calculate and assign all these nodes' heuristics in a batch
        # 3. Sort array into existing queue_list

        # Step 1:
        print("Getting branches...")
        depth = 4  # Depth of branches
        all_nodes = self.get_all_next_states(cubestate)
        final_nodes = []
        for d in range(depth-1):
            new_nodes = []
            for n in all_nodes:
                steps = self.get_all_next_states(n)
                for s in steps:
                    new_nodes.append(s)
            for new in new_nodes:
                all_nodes.append(new)
                if d == depth-2:
                    final_nodes.append(new)


        # Step 2:
        print("Calculating and setting heuristics...")
        state_lists = []
        for n in all_nodes:
            state_lists.append(n.encoded_data)
        batch_heuristics = self.value_model.get_state_value(state_lists)
        for i in range(len(all_nodes)):
            h = batch_heuristics[i].item()
            all_nodes[i].set_total_cost(self.get_total_cost(all_nodes[i], h))

        # Step 3:
        print("Sorting final nodes into queue...")
        for f in final_nodes:
            insert_index = len(queue_list)
            for i in range(len(queue_list)):
                if f.total_cost < queue_list[i].total_cost:
                    insert_index = i
                    break
            queue_list.insert(insert_index, f)

    def get_total_cost(self, state, h):
        # The A* total cost of this "node" (cubestate). Weight parameter is tunable.
        g = state.g_cost
        return g + (1.25 * h)

    def get_state_heuristic(self, state):
        return self.value_model.get_state_value(state.encoded_data)
