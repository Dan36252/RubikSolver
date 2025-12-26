# In this code, the goal is to train the "Main Rubik's Cube Algo" - one that takes a cube state and outputs the next best move.

# Using PyTorch; Keras is weird >:(
import torch, time
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import re
from CubeState import MOVE_SEQUENCE
from accelerate import Accelerator
from datetime import datetime
from ModelClass import RubikNN, device

#torch.set_num_threads(8)
#torch.set_num_interop_threads(1)

# Reading Data from File
print("===================== Setting Up Data =====================")
from DataIO import load_data

print(f"Using {device} device")

X_train, Y_train = load_data()

# Train-Test split
test_size = 20000
X_test = X_train[-test_size:]
Y_test = Y_train[-test_size:]
X_train = X_train[:-test_size]
Y_train = Y_train[:-test_size]

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

def X_transform(X):
    return torch.tensor((X-3)/3)

def Y_transform(y):
    return torch.cat((torch.zeros(20, dtype=torch.float16).scatter_(dim=0, index=torch.tensor(y[0], dtype=torch.int32), value=1), torch.tensor([((y[1]+32)/32)], dtype=torch.float16)), dim=0)

train_dataset = CubeStatesDataset(
    X_train,
    Y_train,
    transform=Lambda(X_transform),
    target_transform=Lambda(Y_transform)
)

test_dataset = CubeStatesDataset(
    X_test,
    Y_test,
    transform=Lambda(X_transform),
    target_transform=Lambda(Y_transform)
)
print("Success!")

# Test the datasets!
print("===================== Testing Datasets =====================")
print(f"Train row index 1: {train_dataset[1]}")
print(f"Test row index 1: {test_dataset[1]}")
print("Praise God!")

# Create Data Loaders
from torch.utils.data import DataLoader

batch_factor = 4
batch_size = 64*batch_factor

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Creating Neural Net!!!
print("===================== Creating NN Object =====================")
model = RubikNN()
#torch.compile(model)
model = model.to(device)
#torch.compile(model)
model.load_state_dict(torch.load('model_checkpoint_weights3.pth', weights_only=True), strict=False)
print(model)

accelerator = Accelerator()

# TRAINING !!!! :O
print("===================== Training =====================")

# Tuning parameters
learning_rate = 2e-5
epochs = 200

# Custom learning rate function
def learning_rate_fun(epoch):
    if epoch < epochs/2:
        return 1.0
    else:
        return 5/(5*(epoch-(epochs/2-1)))

# Initialize the loss function
#loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.MSELoss() # TODO: Make custom Loss function that uses float32 not 64
def loss_fn(output, target):
    return torch.mean((output-target)**2)

# Initialize Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_fun)

# Setup Accelerator
#model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

from datetime import datetime
time_stamp = datetime.now().__repr__()
log_string = ["Beginning of Training Log File\n"]

def train_loop(dataloader, model, loss_fn, optimizer, log_string):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    t0 = time.time()
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

        if batch % (1000/batch_factor) == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            t1000 = time.time()-t0
            log_str = f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time: {t1000}"
            log_string[0] += log_str+'\n'
            print(log_str)
            t0 = time.time()


def test_loop(dataloader, model, loss_fn, log_string):
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
            X = X.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            pred = model(X)
            sliced_pred = torch.from_numpy(np.zeros((len(pred), len(pred[0])-1), dtype=np.float32))
            sliced_y = torch.from_numpy(np.zeros((len(y), len(y[0])-1), dtype=np.float32))

            for i in range(len(y)):
                sliced_pred[i] = pred[i][:-1]
                sliced_y[i] = y[i][:-1]

            #print(f"X shape: {X.shape}, pred shape: {pred.shape}, y shape: {y.shape}")
            test_loss += loss_fn(pred, y).item()
            correct += (sliced_pred.argmax(1) == sliced_y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    eval_log = f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    log_string[0] += eval_log+'\n'
    print(eval_log)

def update_training_logs(log_string):
    # CALL ONCE PER EPOCH AFTER TRAINING LOOP
    # Update the learning rate scheduler
    #lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    cur_lr_log = f"Current learning rate: {current_lr}"
    log_string[0] += cur_lr_log+'\n'
    print(cur_lr_log)


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"training_log_{timestamp}.txt"
# Train-test loop!!!
for e in range(epochs):
    epoch_log = f"[------------- EPOCH {e+1} -------------]"
    log_string[0] += epoch_log+'\n'
    print(epoch_log)
    train_loop(train_dataloader, model, loss_fn, optimizer, log_string)
    test_loop(test_dataloader, model, loss_fn, log_string)
    update_training_logs(log_string)
    # Save model checkpoint
    torch.save(model.state_dict(), time_stamp+'_model_weights.pth')

    # Generate a timestamp string (e.g., 2023-10-27_14-30-05)
    with open(filename, 'a') as file:
        file.write(log_string[0])
    log_string[0] = ""

print("Done :)")

# Saving Model
print("================ Saving Model ================")
torch.save(model, time_stamp+'_model.pth')
# model = torch.load('model.pth', weights_only=False)