import torch
import torch.nn as nn
import numpy as np
import cv2
from Code.Vision.VisionModelClass import VisionCNN, OldVisionNN, device, X_transform, Y_transform, IMG_WIDTH, IMG_HEIGHT
from Code.Data.VisionDataIO import reshape_channels_first
from numpy.ma.core import reshape


def load_model(weights_path="LatestCNNVisionWeights.pth"):#"WorkingVisionWeights.pth"):
    print(f"DEVICE: {device}")
    model = VisionCNN().to(device)
    print(model)
    try:
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device), strict=False)
    except:
        model.load_state_dict(torch.load("src/"+weights_path, weights_only=True, map_location=device), strict=False)
    return model

class Model:
    def __init__(self, use_cnn=True):
        #super(Model, self).__init__()
        self.model = load_model("CNN-Vision-3-8-Weights.pth" if use_cnn else "WorkingVisionWeights.pth")
        self.use_cnn = use_cnn

    def predict_probs(self, correct_col_img):
        self.model.eval()
        img = cv2.resize(correct_col_img, (IMG_WIDTH, IMG_HEIGHT))
        if self.use_cnn:
            img = reshape_channels_first(img)
            img = torch.from_numpy(img.astype(np.float32)).to(device)
        else:
            img = torch.from_numpy(img.flatten().astype(np.float32)).to(device)

        print("IMAGE SHAPE:")
        print(img.shape)
        logits = self.model.forward(img)
        color_probs = []
        for i in range(9):
            color_logits = logits[i * 6:i * 6 + 6]
            color_probs.append(color_logits.tolist())
        print("MODEL PREDICTED PROBABILITIES:")
        print(color_probs)
        # Returns a 9x6 nested list of color probabilities for each sticker
        return color_probs

    def predict_strict(self, correct_col_img):
        self.model.eval()
        img = cv2.resize(correct_col_img, (IMG_WIDTH, IMG_HEIGHT))
        if self.use_cnn:
            img = reshape_channels_first(img)
            img = torch.from_numpy(img.astype(np.float32)).to(device)
        else:
            img = torch.from_numpy(img.flatten().astype(np.float32)).to(device)

        print("IMAGE SHAPE:")
        print(img.shape)
        logits = self.model.forward(img)
        colors = []
        for i in range(9):
            color_logits = logits[i*6:i*6+6]
            colors.append(color_logits.argmax().item())
        print("MODEL PREDICTED COLORS:")
        print(colors)
        return colors

