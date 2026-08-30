import torch
import torch.nn as nn
import numpy as np
import math, copy, random
from Code.Data.CubeState import CubeState, MOVE_SEQUENCE, SOLVED_STATE
from Code.Data.TransformerDataIO import state_matches_mask, F2L_MASK

# Implementation inspired by https://github.com/karpathy/nanoGPT/

CONTEXT_SIZE = 100  # Max length of token sequence that model can accept. Must be at least 73 for F2L Solver

# This model is for solving the F2L of the cube only.
class TransformerDecoder(nn.Module):
    def __init__(self, input_size, output_size, n_blocks=6, d_model=512, n_heads=8):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_blocks = n_blocks
        self.d_model = d_model

        self.embedder = Embedder(input_size, d_model)
        self.pos_encoder = ParallelPositionalEncoding(d_model, CONTEXT_SIZE)
        self.final_layer = nn.Linear(d_model, output_size)
        self.softmax = nn.Softmax(dim=-1)

        self.blocks = nn.Sequential()
        for i in range(n_blocks):
            self.blocks.append(DecoderBlock(d_model, n_heads))

    def forward(self, x):

        x1 = self.embedder(x)
        x2 = self.pos_encoder(x1)
        x3 = self.blocks(x2)
        x4 = self.final_layer(x3)

        return x4

    def cube_to_input(self, cubestate, prev_move):
        # Convert a CubeState object and a prev_move string into an input token tensor

        assert type(cubestate) == CubeState
        assert prev_move in MOVE_SEQUENCE or prev_move is None

        move_one_hot = [0]*len(MOVE_SEQUENCE)
        if not prev_move is None: move_one_hot[MOVE_SEQUENCE.index(prev_move)] = 1

        input_token = torch.tensor(cubestate.encoded_data + move_one_hot)

        return input_token

    def get_sorted_indices(self, tensor):
        # Just like torch.argmax(), but returns an ordered list of indices, based on the descending sorting of the tensor's values
        l = tensor.tolist()
        sorted_indices = []
        for i in range(len(l)):
            largest_val = float('-inf')
            largest_idx = -1
            for j in range(len(l)):
                if j in sorted_indices:
                    continue
                elif l[j] > largest_val:
                    largest_idx = j
                    largest_val = l[j]
            sorted_indices.append(largest_idx)
        return sorted_indices

    def solve(self, orig_cubestate):
        assert type(orig_cubestate) == CubeState
        self.eval()

        cubestate = orig_cubestate.deepcopy()
        final_solution = []
        temp_solution = []

        seen_states = []
        input_tokens = []
        prev_move = None
        max_moves = 200
        move_count = 0
        repeated_count = 0
        max_attempts = 20
        attempt = 0

        while not state_matches_mask(cubestate.flat_data, F2L_MASK) and attempt < max_attempts:
            # Get next input token
            next_input_token = self.cube_to_input(cubestate, prev_move)
            input_tokens.append(next_input_token)

            # Create input tensor
            n_tokens = len(input_tokens)
            n_features = len(input_tokens[0])
            x = torch.stack(input_tokens).reshape(1, n_tokens, n_features)

            # Forward. Call forward() rather than re-inlining the layers, so that inference always
            # runs the exact same path as training (positional encoding included).
            logits = self.softmax(self.forward(x))

            # Interpret output
            next_one_hot = logits[0][-1]
            next_move_index = next_one_hot.argmax()
            next_move = MOVE_SEQUENCE[next_move_index]

            # Perform move on CubeState
            cubestate.move(next_move)
            move_count += 1
            prev_move = next_move

            # Check if this state was already seen, or if max moves reached
            if cubestate.flat_data in seen_states:
                # Backtrack
                cubestate.move(CubeState.get_reverse_move(prev_move))

                # Find next most probable move that results in an unseen state
                possible_moves = self.get_sorted_indices(next_one_hot)
                for i in range(len(possible_moves)-1):
                    trying_move = MOVE_SEQUENCE[possible_moves[i+1]]
                    possible_next_state = cubestate.move(trying_move, just_trying=True)
                    if possible_next_state in seen_states:
                        continue
                    else:
                        next_move = trying_move
                        break

                # Apply and save this move
                cubestate.move(next_move)
                prev_move = next_move

                repeated_count += 1

            elif move_count >= max_moves:
                # Move the original cube once randomly and save the move
                new_move = MOVE_SEQUENCE[random.randint(1, len(MOVE_SEQUENCE)-2)]
                orig_cubestate.move(new_move)
                cubestate = orig_cubestate.deepcopy()
                final_solution.append(new_move)

                # Start a new attempt
                temp_solution = []
                seen_states = []
                input_tokens = []
                prev_move = None
                move_count = 0
                repeated_count = 0
                attempt += 1
            else:
                # Solution looking good so far; save current cubestate to seen_states, and save this move in temp_solution
                seen_states.append(cubestate.flat_data.copy())
                temp_solution.append(prev_move)

        # Check if cube was solved
        if state_matches_mask(cubestate.flat_data, F2L_MASK):
            print(f"Praise God, F2L solved!!! Attempts: {attempt}, Seen States: {len(seen_states)}, Repeated States: {repeated_count}")
            final_solution = final_solution + temp_solution
            return final_solution
        else:
            print(f"Failed to solve cube. Attempts: {attempt}, Seen States: {len(seen_states)}, Repeated States: {repeated_count}")
            return None


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.linear2 = nn.Linear(d_model*4, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pre-norm residuals: each skip connection carries the un-normalized input forward, so the
        # residual stream stays an unbroken identity path from the first block to the last.
        x1 = self.layer_norm1(x)
        x2 = x + self.self_attention(x1, True)

        x3 = self.layer_norm2(x2)
        x4 = self.linear1(x3)
        x5 = self.relu(x4)
        x6 = x2 + self.linear2(x5)

        return x6


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        # d_model is the size of the input tokens (number of features)
        self.d_model = d_model
        # n_heads is the number of heads. must be a factor of d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        # d_k is the size of the projected queries, keys, and values which undergo dot-products and attention weight combination.
        self.d_k = d_model / n_heads
        self.weight_factor = math.sqrt(self.d_k)

        # the linear transformation that simultaneously prepares queries, keys, and values from the input tokens
        self.qkv_projection = nn.Linear(d_model, 3*d_model)
        # the attention weight mask applied when training
        self.register_buffer("attention_mask", torch.tril(torch.ones((CONTEXT_SIZE, CONTEXT_SIZE))).view(1, 1, CONTEXT_SIZE, CONTEXT_SIZE))


    def forward(self, x, masked=True):
        # The expected input for the multihead attention is shaped (batches, tokens, features). features = d_model
        n_batches, n_tokens, n_features = x.shape

        # First step is to generate queries, keys, and values from the input, and split each into separate heads
        queries, keys, values = torch.tensor_split(self.qkv_projection(x), sections=3, dim=-1)
        queries = queries.view(n_batches, n_tokens, self.n_heads, int(n_features / self.n_heads)).transpose(1, 2)
        keys = keys.view(n_batches, n_tokens, self.n_heads, int(n_features / self.n_heads)).transpose(1, 2)
        values = values.view(n_batches, n_tokens, self.n_heads, int(n_features / self.n_heads)).transpose(1, 2)
        # now, q k and v have shape (batches, heads, tokens, features)

        # Apply dot-product between each query-key pair, storing results in a matrix
        attention_weights = torch.matmul(queries, torch.transpose(keys, -1, -2)) / self.weight_factor

        # If we're masking the attention to prevent tokens from attending to future tokens, apply a mask to attention_weights (masked attention)
        if masked: attention_weights = attention_weights.masked_fill(self.attention_mask[:, :, :n_tokens, :n_tokens] == 0, float('-inf'))

        # Normalize the weights corresponding to each query using softmax
        attention_weights = torch.softmax(attention_weights, -1)

        # Apply the calculated weights to the values for each query, storing results in the output matrix
        output = torch.matmul(attention_weights, values)

        # Get the output vector for each token by concatenating the corresponding outputs from all the different heads
        output = output.transpose(1, 2).reshape(n_batches, n_tokens, n_features)

        return output

# parallel positional encoding code was AI generated
class ParallelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 150):
        super().__init__()

        # 1. Create a static matrix of shape [max_len, d_model] to hold the encodings
        pe = torch.zeros(max_len, d_model)

        # 2. Generate a column vector for positions: shape [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 3. Generate a row vector for the geometric progression of frequencies: shape [d_model / 2]
        # We only compute for half the dimension because we pair sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # 4. Parallel broadcasting: Multiply the position column by the frequency row
        # This yields a [max_len, d_model / 2] matrix instantly via GPU vectorization
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        # 5. Reshape to [1, max_len, d_model] for smooth batch broadcasting later
        pe = pe.unsqueeze(0)

        # Register as a buffer so it moves to the GPU with the model but isn't trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        # The addition is also fully parallelized across all batches, tokens, and dimensions
        return x + self.pe[:, :x.size(1)]


class Embedder(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.linear = nn.Linear(input_size, d_model)

    def forward(self, x):
        projected = self.linear(x)
        return projected


