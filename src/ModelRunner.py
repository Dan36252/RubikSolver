import torch
import torch.nn as nn
import numpy as np
from DataIO import move_code_to_3_bits
from CubeState import MOVE_SEQUENCE
from ModelClass import RubikNN, device

def load_model(weights_path='model_checkpoint_weights3.pth'):
    model = RubikNN().to(device)
    model.load_state_dict(torch.load(weights_path, weights_only=True), strict=False)
    return model

def transform_X(x):
    return torch.from_numpy(((x - 3) / 3).astype(np.float32)).to(device)

def transform_moves_list(moves):
    transformed = []

    # Fill list with 3-number versions of move codes, in reverse order
    for i in range(len(moves)):
        three_num = move_code_to_3_bits(MOVE_SEQUENCE.index(moves[i]))
        for j in range(len(three_num)):
            transformed.insert(0, three_num[len(three_num)-j-1])

    # Fill rest of list with 0s
    for s in range(90 - len(transformed)):
        transformed.append(0)

    return torch.tensor(transformed[:90], dtype=torch.float32).to(device)

class Model:
    def __init__(self):
        self.model = load_model()

    # def Y_transform(self, y):
    #     return torch.cat((torch.zeros(20, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y[0], dtype=torch.int32), value=1), torch.tensor([((y[1] + 32) / 32)])), dim=0)

    def predict(self, state, prev_moves):
        # state = numpy list of shape (54,)
        # prev_moves = list of Move Letters (R', U2, etc.). NOT in reverse order. (1st move, 2nd, ...)
        state = transform_X(state)
        prev_moves = transform_moves_list(prev_moves)
        # print("Calculated Prev Moves:")
        # print(prev_moves)
        X = torch.cat((state, prev_moves))
        self.model.eval()
        logits = self.model.forward(X)
        pred = MOVE_SEQUENCE[logits[:-1].argmax()]
        # print("Prediction:")
        # print(pred)
        return pred


