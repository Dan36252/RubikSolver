# Trains the custom LSTM SolverRNN to solve a Rubik's cube one move at a time.
#
# The raw data from load_data() is a FLAT stream of (cube_state, move) pairs
# across many solutions, concatenated end to end. Each solution ends with the
# '#' stop token. We split that stream back into variable-length sequences at
# every '#', pad each batch to its longest sequence, and ignore the padded
# steps in the loss (CrossEntropyLoss ignore_index = -100).

import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from Code.Data.DataIO import load_data
from Code.Data.CubeState import CubeState, MOVE_SEQUENCE, SOLVED_STATE
from Code.Solver.RNNModelClass import SolverRNN, StagedStickerMask, device

# Token indices in MOVE_SEQUENCE.
STOP_INDEX = MOVE_SEQUENCE.index('#')   # marks the end of one solution
PAD_INDEX = -100                        # CrossEntropyLoss's default ignore_index


# --------------------------------------------------------------------------- #
# Data pipeline: flat (state, move) stream  ->  variable-length sequences
# --------------------------------------------------------------------------- #

def split_into_sequences(X, Y):
    """
    Split the flat (state, one-hot-move) stream into a list of solution
    sequences. A new sequence is cut after every '#' (stop) move, which is
    included as the final step of its sequence.

    Args:
        X: (N, 54) array of cube states.
        Y: (N, 20) array of one-hot moves.

    Returns:
        list of (states, moves) tuples, where
            states is (T_i, 54) float32 and
            moves  is (T_i,)    int64 move indices.
    """
    move_indices = Y.argmax(axis=1)  # one-hot -> class index

    sequences = []
    cur_states, cur_moves = [], []
    for i in range(len(move_indices)):
        cur_states.append(X[i])
        cur_moves.append(move_indices[i])
        if move_indices[i] == STOP_INDEX:
            sequences.append((
                np.asarray(cur_states, dtype=np.float32),
                np.asarray(cur_moves, dtype=np.int64),
            ))
            cur_states, cur_moves = [], []

    # Keep a trailing partial sequence (a solution with no closing '#'), if any.
    if cur_states:
        sequences.append((
            np.asarray(cur_states, dtype=np.float32),
            np.asarray(cur_moves, dtype=np.int64),
        ))

    return sequences


class CubeSequenceDataset(Dataset):
    """
    Holds one solution sequence per item. Raw cube states are stored, and the
    staged sticker mask is applied on access — a fresh StagedStickerMask per
    sequence, walking the states in solve order so the stage advances correctly.
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        states, moves = self.sequences[idx]   # states: (T, 54) raw color codes
        mask = StagedStickerMask()
        masked = np.asarray([mask(states[t]) for t in range(states.shape[0])],
                            dtype=np.float32)
        return torch.from_numpy(masked), torch.from_numpy(moves)


def collate_sequences(batch):
    """
    Pad a batch of variable-length sequences to the batch's longest sequence.

    States are right-padded with zeros; targets are right-padded with PAD_INDEX
    (-100) so the padded steps are ignored by CrossEntropyLoss. Right-padding
    keeps every real step before any pad step, so the model's output feedback
    never leaks from a padded step into a real one.

    Returns:
        padded_states: (B, T_max, 54) float32
        padded_moves:  (B, T_max)     int64  (pad positions == PAD_INDEX)
        lengths:       (B,)           int64  true length of each sequence
    """
    states = [b[0] for b in batch]
    moves = [b[1] for b in batch]
    lengths = torch.tensor([s.shape[0] for s in states], dtype=torch.int64)

    padded_states = pad_sequence(states, batch_first=True, padding_value=0.0)
    padded_moves = pad_sequence(moves, batch_first=True, padding_value=PAD_INDEX)

    return padded_states, padded_moves, lengths


# --------------------------------------------------------------------------- #
# Train / test loops
# --------------------------------------------------------------------------- #

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    num_batches = len(dataloader)
    t0 = time.time()

    for batch, (X, y, _lengths) in enumerate(dataloader):
        X = X.to(device, dtype=torch.float32)   # already staged-masked by the dataset
        y = y.to(device, dtype=torch.long)

        # Teacher-forced logits for every step: (B, T_max, output_size).
        logits = model.train_forward(X, target_seq=y)

        # Option A: flatten and let ignore_index drop the padded steps.
        loss = loss_fn(logits.reshape(-1, model.output_size), y.reshape(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            print(f"  loss: {loss.item():>7f}  [{batch:>4d}/{num_batches:>4d}]  "
                  f"time: {time.time() - t0:.1f}s")
            t0 = time.time()


def test_loop(dataloader, model, loss_fn):
    """Reports masked per-move accuracy and average loss (teacher-forced)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y, _lengths in dataloader:
            X = X.to(device, dtype=torch.float32)   # already staged-masked by the dataset
            y = y.to(device, dtype=torch.long)

            logits = model.train_forward(X, target_seq=y)
            total_loss += loss_fn(logits.reshape(-1, model.output_size), y.reshape(-1)).item()

            preds = logits.argmax(dim=-1)
            mask = (y != PAD_INDEX)            # ignore padded steps
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = 100.0 * correct / max(total, 1)
    print(f"Test:  per-move accuracy: {acc:>0.1f}%   avg loss: {avg_loss:>8f}")
    return acc, avg_loss


def evaluate_solve_rate(model, sequences, sample=50, max_moves=200):
    """
    End-to-end check: let the model drive a real cube from each sequence's
    starting (scrambled) state and report how many it actually solves within
    `max_moves`. This exercises the autoregressive forward() loop, unlike the
    teacher-forced accuracy above.
    """
    model.eval()
    n = min(sample, len(sequences))
    solved = 0

    for i in range(n):
        start_state = sequences[i][0][0]                  # first state of the sequence
        start_list = [int(round(float(v))) for v in start_state]

        moves = model.forward(torch.tensor(start_state, dtype=torch.float32),
                              max_moves=max_moves)

        # Replay the predicted moves to confirm the cube is genuinely solved.
        cube = CubeState(start_list)
        for mv in moves:
            cube.move(mv)
        if cube.get_flat_data() == SOLVED_STATE:
            solved += 1

    print(f"Solve rate: {solved}/{n} scrambles solved within {max_moves} moves "
          f"({100.0 * solved / max(n, 1):>0.1f}%)")
    return solved, n


# --------------------------------------------------------------------------- #
# Main script
# --------------------------------------------------------------------------- #

def main():
    print(f"Using {device} device")

    print("===================== Loading Data =====================")
    X_train, Y_train = load_data(
        processed_data_path="Data/Solver/ProcessedPlain",
        encode=False,
        output_type="move",
        include_prev_moves_input=False,
    )
    print(f"Loaded flat stream: X={X_train.shape}, Y={Y_train.shape}")

    print("===================== Building Sequences =====================")
    sequences = split_into_sequences(X_train, Y_train)
    print(f"Split into {len(sequences)} solution sequences.")
    if len(sequences) == 0:
        raise RuntimeError("No sequences were produced — check the data / stop token.")

    # Train/test split over whole sequences (not individual moves).
    test_size = max(1, int(len(sequences) * 0.05))
    train_sequences = sequences[:-test_size]
    test_sequences = sequences[-test_size:]
    print(f"Train sequences: {len(train_sequences)}   Test sequences: {len(test_sequences)}")

    train_dataset = CubeSequenceDataset(train_sequences)
    test_dataset = CubeSequenceDataset(test_sequences)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_sequences)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_sequences)

    print("===================== Building Model =====================")
    model = SolverRNN(input_size=54, hidden_size=1080, hidden_layers=3).to(device)
    print(model)

    learning_rate = 1e-4
    epochs = 20
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)   # Option A
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("===================== Training =====================")
    for e in range(epochs):
        print(f"[------------- EPOCH {e + 1}/{epochs} -------------]")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
        torch.save(model.state_dict(), "SolverRNN_weights.pth")

    print("===================== Final Testing =====================")
    test_loop(test_dataloader, model, loss_fn)
    evaluate_solve_rate(model, test_sequences, sample=50, max_moves=200)

    print("===================== Saving Model =====================")
    torch.save(model.state_dict(), "SolverRNN_weights.pth")
    print("Done :)")


if __name__ == "__main__":
    main()
