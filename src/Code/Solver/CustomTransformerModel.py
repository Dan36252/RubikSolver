import torch
import torch.nn as nn
import math

# Implementation inspired by https://github.com/karpathy/nanoGPT/

CONTEXT_SIZE = 100  # Max length of token sequence that model can accept. Must be at least 73 for F2L Solver

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, output_size, n_blocks=4, d_model=512):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_blocks = n_blocks
        self.d_model = d_model

    def train_forward(self):
        print("IMPLEMENT train_forward()!")

class multihead_attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        # d_model is the size of the input tokens
        self.d_model = d_model
        # n_heads is the number of heads. must be a factor of d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        # d_k is the size of the projected queries, keys, and values which undergo dot-products and attention weight combination.
        self.d_k = d_model / n_heads

        # the linear transformation that simultaneously prepares queries, keys, and values from the input tokens
        self.qkv_projection = nn.Linear(d_model, 3*d_model)
        # the attention weight mask applied when training
        self.register_buffer("attention_mask", torch.tril(torch.ones((CONTEXT_SIZE, CONTEXT_SIZE))).view(1, 1, CONTEXT_SIZE, CONTEXT_SIZE))


    def forward(self, x, masked=False):
        # The expected input for the multihead attention is shaped (batches, tokens, features). features = d_model
        n_batches, n_tokens, n_features = x.shape

        # First step is to generate queries, keys, and values from the input, and split each into separate heads
        queries, keys, values = torch.tensor_split(self.qkv_projection(x), sections=3)
        queries = queries.view(n_batches, n_tokens, self.n_heads, n_features / self.n_heads).transpose(1, 2)
        keys = keys.view(n_batches, n_tokens, self.n_heads, n_features / self.n_heads).transpose(1, 2)
        values = values.view(n_batches, n_tokens, self.n_heads, n_features / self.n_heads).transpose(1, 2)
        # now, q k and v have shape (batches, heads, tokens, features)

        # Apply dot-product between each query-key pair, storing results in a matrix
        attention_weights = torch.matmul(queries, torch.transpose(keys, -1, -2)) / self.d_k

        # If we're masking the attention to prevent tokens from attending to future tokens, apply a mask to attention_weights (masked attention)
        if masked: attention_weights = attention_weights.masked_fill(self.attention_mask[:, :, :n_tokens, :n_tokens] == 0, float('-inf'))

        # Normalize the weights corresponding to each query using softmax
        attention_weights = torch.softmax(attention_weights, -1)

        # Apply the calculated weights to the values for each query, storing results in the output matrix
        output = torch.matmul(attention_weights, values)

        return output



