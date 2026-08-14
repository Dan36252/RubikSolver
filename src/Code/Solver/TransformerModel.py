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

    def train_forward(self):
        print("IMPLEMENT train_forward()!")

class self_attention(nn.Module):
    def __init__(self, dimension, masked=False):
        super().__init__()

        self.dimension = dimension
        self.masked = masked

        self.weight_scale_factor = math.sqrt(dimension)

    def forward(self, queries, keys, values):
        # Apply dot-product between each query-key pair, storing results in a matrix
        attention_weights = torch.matmul(queries, torch.transpose(keys, 0, 1)) / self.weight_scale_factor

        # Normalize the weights corresponding to each query using softmax
        attention_weights = torch.softmax(attention_weights, 1)

        # Apply the calculated weights to the values for each query, storing results in the output matrix
        output = torch.matmul(attention_weights, values)

        return output

