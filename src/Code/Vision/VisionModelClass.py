import torch
from torch import nn

print(f"CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
IMG_WIDTH = 24
IMG_HEIGHT = 24

def X_transform(X):
    return torch.tensor(X/255)

def Y_transform(y):
    #return torch.cat((torch.zeros(20, dtype=torch.float16).scatter_(dim=0, index=torch.tensor(y[0], dtype=torch.int32), value=1), torch.tensor([((y[1]+32)/32)], dtype=torch.float16)), dim=0)
    result = torch.empty(0)
    for i in range(9):
        color_one_hot = torch.zeros(6).scatter_(dim=0, index=torch.tensor(y[i], dtype=torch.int32), value=1)
        result = torch.cat((result, color_one_hot))
    return result

class MyLambda(nn.Module):
    """ Input: A Function. Returns: A Module that can be used inside nn.Sequential """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def PrintInputSize(x):
    print(f"Layer shape: {x.shape}")
    return x

def StoreInputSize(x):
    print(f"This layer Input size is: {len(x)}")
    return len(x)

class VisionCNN(nn.Module):  # TODO: Ready to try training new model!

    def __init__(self, im_width=IMG_WIDTH, im_height=IMG_HEIGHT):
        super().__init__()
        self.input_shape_len = 0
        #self.last_input_size = 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=36, kernel_size=6, padding=0, stride=2), # Output = 21x21
            nn.ReLU(),
            nn.MaxPool2d(2),
            #MyLambda(PrintInputSize),
            nn.Flatten(self.input_shape_len-3, self.input_shape_len-1),
            #MyLambda(self.store_input_size),
            MyLambda(PrintInputSize),
            nn.Linear(25*36, 1024),
            #MyLambda(PrintInputSize),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 54)
        )

    def store_input_size(self, x):
        print(f"Input size is: {len(x)}")
        self.last_input_size = len(x)
        return x

    def forward(self, x):
        self.input_shape_len = len(x.shape)
        #print(self.input_shape_len-3, self.input_shape_len-1)
        logits = self.layers(x)
        return logits

class OldVisionCNN(nn.Module):
    def __init__(self, im_width=IMG_WIDTH, im_height=IMG_HEIGHT):
        super().__init__()
        self.input_shape_len = 0
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=9, out_channels=18, kernel_size=4, padding="same"),
            nn.ReLU(),
            #MyLambda(PrintInputSize),
            nn.Flatten(self.input_shape_len-3, self.input_shape_len-1),
            #MyLambda(PrintInputSize),
            nn.Linear(int(18*im_width*im_height/4), 128),
            #MyLambda(PrintInputSize),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 54)
        )

    def forward(self, x):
        self.input_shape_len = len(x.shape)
        #print(self.input_shape_len-3, self.input_shape_len-1)
        logits = self.layers(x)
        return logits

class OldVisionNN(nn.Module):
    # ASSUMING: Input images are in Full HSV format.
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Input: 1728 --> 24 x 24 x 2 array but flattened (HSV cube face image, without the V)
            # Input is standardized. [between 0 & 1]
            nn.Linear(IMG_WIDTH*IMG_HEIGHT*3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 54),
            # Output: 9 sticker colors, but times 6 because each color is one-hot encoded.
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits