import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

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