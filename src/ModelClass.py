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

class VisionCNN(nn.Module):
    def __init__(self, im_width, im_height):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6*im_width*im_height, 54),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

class OldVisionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Input: 1728 --> 24 x 24 x 3 array but flattened (RGB cube face image)
            # Input is standardized. [between 0 & 1]
            nn.Linear(1728, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 54),
            # Output: 9 sticker colors, x 6 because each color is one-hot encoded.
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits