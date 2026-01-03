import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
from torch import nn
from PIL import Image
import cv2
import numpy as np
import re
from pathlib import Path
import time, random
from ModelClass import VisionCNN, device
from VisionDataIO import load_data

print("===================== Loading Data =====================")

IM_WIDTH = 24
IM_HEIGHT = 24

#create_dataset_txt("RubikFaceData", "RubikFaceDataLabels.txt")
#X, y = get_dataset("RubikFaceData.txt", "RubikFaceDataLabels.txt")
X, y = load_data()

test_size = 20
X_test = X[-test_size:]
Y_test = y[-test_size:]
X_train = X[:-test_size]
Y_train = y[:-test_size]

#X2, y2 = generate_data(10000, red=[175, 45, 45], yellow=[200, 70, 70], green=[100, 200, 90], white=[190, 200, 200], orange=[220, 100, 40], blue=[25, 95, 150])
# print("Generated Image:")
# print(X2[0])
# print("Generated Label:")
# print(y2[0])
# cv2.imshow("test", np.array(X2[0], dtype=np.uint8).reshape((24, 24, 3)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# X_train = np.concat((X_train, X2))
# Y_train = np.concat((Y_train, y2))

print(f"Train X: {len(X_train)}")
print(f"Train Y: {len(Y_train)}")
print(f"Test X: {len(X_test)}")
print(f"Test Y: {len(Y_test)}")


print("===================== Creating Datasets =====================")
class RubikFaceDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        face = self.X[idx]
        code = self.y[idx]
        if self.transform:
            face = self.transform(face)
        if self.target_transform:
            code = self.target_transform(code)
        return face, code

def X_transform(X):
    return torch.tensor(X/255)

def Y_transform(y):
    #return torch.cat((torch.zeros(20, dtype=torch.float16).scatter_(dim=0, index=torch.tensor(y[0], dtype=torch.int32), value=1), torch.tensor([((y[1]+32)/32)], dtype=torch.float16)), dim=0)
    result = torch.empty(0)
    for i in range(9):
        color_one_hot = torch.zeros(6).scatter_(dim=0, index=torch.tensor(y[i], dtype=torch.int32), value=1)
        result = torch.cat((result, color_one_hot))
    return result

train_dataset = RubikFaceDataset(
    X_train,
    Y_train,
    transform=Lambda(X_transform),
    target_transform=Lambda(Y_transform)
)

test_dataset = RubikFaceDataset(
    X_test,
    Y_test,
    transform=Lambda(X_transform),
    target_transform=Lambda(Y_transform)
)

print(train_dataset[0][1])
print(len(test_dataset))
#print("Success!")

# Create Data Loaders
from torch.utils.data import DataLoader

batch_factor = 1
batch_size = 32*batch_factor

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print("===================== Creating Model =====================")
model = VisionCNN(IM_WIDTH, IM_HEIGHT)
model = model.to(device)

print("===================== Training =====================")

# Tuning parameters
learning_rate = 2e-4
epochs = 200

# Initialize Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Get the current date and time
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H;%M")
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

        if batch % (100/batch_factor) == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            t1 = time.time()-t0
            log_str = f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time: {t1}"
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
            # print("y:")
            # print(y)
            # print("pred:")
            # print(pred)
            # sliced_pred = torch.from_numpy(np.zeros((len(pred), len(pred[0])-1), dtype=np.float32))
            # sliced_y = torch.from_numpy(np.zeros((len(y), len(y[0])-1), dtype=np.float32))
            #
            # for i in range(len(y)):
            #     sliced_pred[i] = pred[i][:-1]
            #     sliced_y[i] = y[i][:-1]

            #print(f"X shape: {X.shape}, pred shape: {pred.shape}, y shape: {y.shape}")
            test_loss += loss_fn(pred, y).item()
            # TODO: Write custom # correct function (there are 9 classes to predict in each output, not one)
            correct = 0
            print(len(pred))
            for i in range(len(pred)):
                for j in range(9):
                    logits = pred[i][6*j:6*j+6]
                    #print(f"logit len: {len(logits)}")
                    if logits.argmax() == y[i][j]:
                        correct += 1
            #correct += (sliced_pred.argmax(1) == sliced_y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= (size*9)
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


filename = f"training_vision_log_{timestamp}.txt"
# Train-test loop!!!
for e in range(epochs):
    epoch_log = f"[------------- EPOCH {e+1} -------------]"
    log_string[0] += epoch_log+'\n'
    print(epoch_log)
    train_loop(train_dataloader, model, loss_fn, optimizer, log_string)
    test_loop(test_dataloader, model, loss_fn, log_string)
    update_training_logs(log_string)
    # Save model checkpoint
    torch.save(model.state_dict(), timestamp+'_vision_model_weights.pth')

    # Generate a timestamp string (e.g., 2023-10-27_14-30-05)
    with open(filename, 'a') as file:
        file.write(log_string[0])
    log_string[0] = ""

print("Done :)")

# Saving Model
print("================ Saving Model ================")
torch.save(model, timestamp+'_vision_model.pth')
# model = torch.load('model.pth', weights_only=False)