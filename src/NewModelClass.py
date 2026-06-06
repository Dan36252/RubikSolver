import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def X_transform(X):
    #return torch.tensor((X-3)/3)
    return X

def Y_transform(y):
    #return torch.tensor(y+20/40)
    return y


class F2LValueNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Input: 54 --> Cube State
            nn.Linear(54, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 320),
            nn.ReLU(),
            nn.Linear(320, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 320),
            nn.ReLU(),
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # Output: 20 different moves (18 + nothing + stop; one-hot), 1 "dist till solution"
            # nn.Linear(256, 21)
            nn.Linear(64, 1)
        )

    def forward(self, x):
        pred = self.layers(x)
        return pred

class EncodedValueNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Input: 54 --> Cube State
            nn.Linear(104, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        pred = self.layers(x)
        return pred