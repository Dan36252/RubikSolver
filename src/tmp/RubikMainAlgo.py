# In this code, the goal is to train the "Main Rubik's Cube Algo" - one that takes a cube state and outputs the next best move.
# TEMP VENV

# Using PyTorch; Keras is weird >:(
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import re
from CubeState import MOVE_SEQUENCE

# Reading Data from File
print("===================== Setting Up Data =====================")
from DataIO import read_cubestates_file, write_cubestates_file, load_processed_data

X_train, Y_train = load_processed_data('ProcessedData/processed_data.txt')
print("Loaded file 0")
X_train1, Y_train1 = load_processed_data('ProcessedData/processed_data1.txt')
X_train = np.concatenate((X_train, X_train1))
Y_train = np.concatenate((Y_train, Y_train1))
print("Loaded file 1")
X_train1, Y_train1 = load_processed_data('ProcessedData/processed_data2.txt')
X_train = np.concatenate((X_train, X_train1))
Y_train = np.concatenate((Y_train, Y_train1))
print("Loaded file 2")
X_train1, Y_train1 = load_processed_data('ProcessedData/processed_data3.txt')
X_train = np.concatenate((X_train, X_train1))
Y_train = np.concatenate((Y_train, Y_train1))
print("Loaded file 3")

# Train-Test split
X_test = X_train[-10000:]
Y_test = Y_train[-10000:]
X_train = X_train[:-10000]
Y_train = Y_train[:-10000]

print(f"Train States: {X_train.shape}")
print(f"Train Moves: {Y_train.shape}")
print(f"Test States: {X_test.shape}")
print(f"Test Moves: {Y_test.shape}")

# Create Datasets
print("===================== Creating Datasets =====================")
class CubeStatesDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        cubestate = self.X[idx]
        move = self.y[idx]
        if self.transform:
            cubestate = self.transform(cubestate)
        if self.target_transform:
            move = self.target_transform(move)
        return cubestate, move

train_dataset = CubeStatesDataset(
    X_train,
    Y_train,
    transform=Lambda(lambda x: torch.from_numpy(((x-3)/3).astype(np.float32))),
    target_transform=Lambda(lambda y: torch.cat((torch.zeros(19, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y[0]), value=1), torch.tensor([((y[1]+32)/32)])), dim=0))
)

test_dataset = CubeStatesDataset(
    X_test,
    Y_test,
    transform=Lambda(lambda x: torch.from_numpy(((x-3)/3).astype(np.float32))),
    target_transform=Lambda(lambda y: torch.cat((torch.zeros(19, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y[0]), value=1), torch.tensor([((y[1]+32)/32)])), dim=0))
)
print("Success!")

# Test the datasets!
print("===================== Testing Datasets =====================")
print(f"Train row index 1: {train_dataset[1]}")
print(f"Test row index 1: {test_dataset[1]}")
print("Praise God!")

# Create Data Loaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Creating Neural Net!!!
print("===================== Creating NN Object =====================")
from torch import nn

class RubikNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(54, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

device = 'cpu'#torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = RubikNN().to(device)
print(model)

# TRAINING !!!! :O
print("===================== Training =====================")

# Tuning parameters
learning_rate = 1e-4
batch_size = 64
epochs = 50

# Custom learning rate function
def learning_rate_fun(epoch):
    if epoch < 25:
        return 1.0
    else:
        return 25/epoch

# Initialize the loss function
#loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.MSELoss() # TODO: Make custom Loss function that uses float32 not 64
def loss_fn(output, target):
    return torch.mean((output-target)**2)

# Initialize Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_fun)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        #print(f"INPUT TYPE: {X.item().dtype}")
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            #print(f"X shape: {X.shape}, pred shape: {pred.shape}, y shape: {y.shape}")
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    checkpoint_log = f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    with open('checkpoint_logs2.txt', 'a') as file:
        file.write(checkpoint_log)
    print(checkpoint_log)

# Train-test loop!!!
for e in range(epochs):
    print(f"[------------- EPOCH {e+1} -------------]")

    # TRAIN
    train_loop(train_dataloader, model, loss_fn, optimizer)

    # Update the learning rate scheduler after the optimizer step
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current lr: {current_lr}")

    # EVALUATE
    test_loop(test_dataloader, model, loss_fn)

    # Save model checkpoint
    torch.save(model.state_dict, 'model_checkpoint_weights2.pth')

print("Done :)")

# Saving Model
print("================ Saving Model ================")
torch.save(model, 'model2.pth')
# model = torch.load('model.pth', weights_only=False)