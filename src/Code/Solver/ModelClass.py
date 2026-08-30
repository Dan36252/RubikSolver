import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def X_transform(X):
    return torch.tensor((X-3)/3)

def Y_transform(y):
    return torch.cat((torch.zeros(20, dtype=torch.float16).scatter_(dim=0, index=torch.tensor(y[0], dtype=torch.int32), value=1), torch.tensor([((y[1]+32)/32)], dtype=torch.float16)), dim=0)


class RubikNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Input: 54 --> Cube State, + 3x30 --> Prev 30 moves (as 3 "tri-bits" each)
            nn.Linear(144, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
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
            nn.Linear(64, 21)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

