import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

device = "cuda" if torch.cuda.is_available() else torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def X_transform(X):
    #return torch.tensor((X-3)/3)
    return X

def Y_transform(y):
    #return torch.tensor(y+20/40)
    return y


class SolverRNN(nn.Module):
    def __init__(self, input_size=54, hidden_size=1080, hidden_layers=3):
        super().__init__()

        # Initialize hidden layers (LSTM cells)
        self.LSTMCells = []
        prev_input_size = input_size
        for i in range(hidden_layers):
            cell = nn.LSTMCell(prev_input_size, hidden_size, bias=True, device=device)
            self.LSTMCells.append(cell)

    def train_forward(self, x):
        # Input 'x' should be
        return None

    def forward(self, x):
        # Input 'x' should be a single cube state of length 54
        return None