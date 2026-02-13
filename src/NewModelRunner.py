import torch
import torch.nn as nn
import numpy as np
from CubeState import MOVE_SEQUENCE
from NewModelClass import F2LValueNN, X_transform, device

def load_model(weights_path='WorkingF2LValueWeights.pth'):
    model = F2LValueNN().to(device)
    model.load_state_dict(torch.load(weights_path, weights_only=True), strict=False)
    return model

class Model:
    def __init__(self):
        self.model = load_model()

    def get_state_value(self, state_list):
        # state_list is a single cube state of length 54, as a list.
        self.model.eval()
        x = np.array(state_list, dtype=np.float32)
        x = X_transform(x)
        x = x.to(device)
        pred = self.model.forward(x)
        return pred.item()

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


