import torch
import torch.nn as nn
import math

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, output_size, n_blocks=4, d_model=512):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_blocks = n_blocks
        self.d_model = d_model

        self.decoder_layer = nn.TransformerDecoderLayer(input_size, 8)

    def train_forward(self):
        print("IMPLEMENT train_forward()!")
