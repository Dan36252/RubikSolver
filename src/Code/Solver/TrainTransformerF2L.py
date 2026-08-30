import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import time, os

from Code.Data.TransformerDataIO import load_f2l_solver_data, raw_data_path
from Code.Data.CubeState import CubeState, MOVE_SEQUENCE
from Code.Solver.TransformerModel import TransformerDecoder, CONTEXT_SIZE

# Training hyperparameters
N_EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

# Model hyperparameters
N_BLOCKS = 6
D_MODEL = 512
N_HEADS = 8

# The batch size is fixed at 1: every batch is exactly one cube solution, shaped (1, tokens, features).
BATCH_SIZE = 1

# Number of cube states held out of training and used only to measure the solve rate.
N_EVAL_CUBES = 100
EVAL_SEED = 0

# MOVE_SEQUENCE[0] is '-', which never appears as a real move in the training data (real moves use
# classes 1-19). The all-zero padding rows written by the data loader therefore argmax to class 0,
# so class 0 doubles as the "this row is padding" label and is ignored by the loss.
PADDING_CLASS = 0

CHECKPOINT_PATH = "Data/Solver/TransformerData/F2LSolver_model.pt"
LOG_EVERY = 500


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_eval_cubestates(global_indices, total_solutions):
    # The processed data holds *encoded* cube states, which cannot be turned back into CubeState
    # objects, but solve() needs real CubeStates. So the starting cube state of each held-out
    # solution is recovered from the raw data files instead.
    #
    # load_f2l_solver_data() concatenates the raw files in sorted filename order with every file
    # contributing the same number of solutions, so a global solution index maps directly onto a
    # (raw file, solution within that file) pair.
    raw_filenames = sorted(os.listdir(raw_data_path))
    n_files = len(raw_filenames)
    assert total_solutions % n_files == 0, f"Cannot map solution indices: {total_solutions} solutions over {n_files} files"
    per_file = total_solutions // n_files

    # Group the wanted solutions by the raw file that contains them
    wanted = {}
    for global_index in global_indices:
        file_index = int(global_index) // per_file
        local_index = int(global_index) % per_file
        wanted.setdefault(file_index, {})[local_index] = None

    for file_index, local_indices in sorted(wanted.items()):
        filename = raw_filenames[file_index]
        last_needed = max(local_indices)
        print(f"  Reading eval cube states from {filename} ({len(local_indices)} wanted)...")

        with open(os.path.join(raw_data_path, filename)) as file:
            solution_index = 0
            is_first_state_of_solution = True
            while solution_index <= last_needed:
                state_line = file.readline()
                move_line = file.readline()
                if not move_line: break

                if is_first_state_of_solution and solution_index in local_indices:
                    state = [int(color) for color in state_line.strip().split()]
                    local_indices[solution_index] = state

                # A '#' move marks the end of a solution, so the next pair starts a new one
                if move_line.strip() == "#":
                    solution_index += 1
                    is_first_state_of_solution = True
                else:
                    is_first_state_of_solution = False

    # Build the CubeStates. f2l_only=True must match how the training data was encoded, so that
    # cube_to_input() produces vectors of the same width the model was trained on.
    cubestates = []
    for local_indices in wanted.values():
        for state in local_indices.values():
            assert state is not None, "Failed to locate an eval solution in the raw data"
            cubestates.append(CubeState(data=state, encode=True, f2l_only=True))
    return cubestates


def build_data():
    # Load every processed F2L data file as one pair of big tensors:
    # states (solutions, tokens, state_features), moves (solutions, tokens, move_features)
    print("Loading F2L solver data...")
    states, moves = load_f2l_solver_data()
    print(f"Loaded states {states.shape}, moves {moves.shape}")

    # The model predicts a single move class per token, so collapse the one-hot move vectors into
    # class indices. Padding rows are all-zero and collapse to PADDING_CLASS.
    targets = moves.argmax(axis=-1).astype(np.int64)
    del moves

    inputs = torch.from_numpy(states)
    targets = torch.from_numpy(targets)
    dataset = TensorDataset(inputs, targets)

    # Hold out N_EVAL_CUBES random solutions; everything else is used for training.
    total_solutions = len(dataset)
    rng = np.random.default_rng(EVAL_SEED)
    eval_indices = rng.choice(total_solutions, size=N_EVAL_CUBES, replace=False)
    eval_lookup = set(int(i) for i in eval_indices)
    train_indices = [i for i in range(total_solutions) if i not in eval_lookup]
    print(f"Split: {len(train_indices)} training solutions, {len(eval_indices)} eval cube states")

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)

    print("Recovering eval cube states from raw data...")
    eval_cubestates = load_eval_cubestates(eval_indices, total_solutions)

    return train_loader, eval_cubestates, inputs.shape[-1], len(MOVE_SEQUENCE), inputs.shape[1]


def evaluate_solve_rate(model, eval_cubestates):
    # Measures what fraction of the held-out cube states the model can actually solve the F2L of,
    # by running the model's own search in solve().
    solved = 0
    failed_invalid_move = 0
    failed_context_limit = 0
    failed_other = 0

    for cubestate in eval_cubestates:
        # solve() mutates the cube state it is given when it restarts an attempt, so hand it a copy
        # and keep the eval set identical across every epoch.
        try:
            solution = model.solve(cubestate.deepcopy())
            if solution is not None: solved += 1
        except KeyError:
            # The model predicted '-' or '#', which are not real cube moves, so MAPS has no entry.
            failed_invalid_move += 1
        except RuntimeError:
            # The token sequence outgrew CONTEXT_SIZE.
            failed_context_limit += 1
        except Exception as error:
            print(f"    Unexpected solve() failure: {type(error).__name__}: {error}")
            failed_other += 1

    solve_rate = solved / len(eval_cubestates)
    print(f"  Solve rate: {solve_rate:.2%} ({solved}/{len(eval_cubestates)})")
    print(f"  Failures - predicted non-move token: {failed_invalid_move}, hit context limit: {failed_context_limit}, other: {failed_other}")
    return solve_rate


def train():
    device = get_device()
    print(f"Training on device: {device}")

    train_loader, eval_cubestates, input_size, output_size, n_tokens = build_data()

    # The causal attention mask and positional encoding are both built at CONTEXT_SIZE, so they can
    # only cover sequences up to that length.
    assert n_tokens <= CONTEXT_SIZE, f"Token count {n_tokens} exceeds model CONTEXT_SIZE {CONTEXT_SIZE}"

    model = TransformerDecoder(input_size=input_size, output_size=output_size, n_blocks=N_BLOCKS, d_model=D_MODEL, n_heads=N_HEADS)
    model.to(device)

    loss_function = nn.CrossEntropyLoss(ignore_index=PADDING_CLASS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    n_batches = len(train_loader)
    for epoch in range(N_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{N_EPOCHS} ---")
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_counted = 0

        for batch_index, (x, y) in enumerate(train_loader):
            # x: (1, tokens, state_features) - the shape TransformerDecoder.forward() expects.
            # y: (1, tokens) - one move class per token.
            x = x.to(device).float()
            y = y.to(device)

            logits = model(x)

            # CrossEntropyLoss wants (rows, classes) predictions against (rows,) targets, so flatten
            # the batch and token dimensions together.
            loss = loss_function(logits.reshape(-1, output_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss, and accuracy over the real (non-padding) tokens only
            running_loss += loss.item()
            with torch.no_grad():
                predictions = logits.reshape(-1, output_size).argmax(dim=-1)
                flat_targets = y.reshape(-1)
                real_tokens = flat_targets != PADDING_CLASS
                running_correct += (predictions[real_tokens] == flat_targets[real_tokens]).sum().item()
                running_counted += int(real_tokens.sum().item())

            if (batch_index + 1) % LOG_EVERY == 0:
                mean_loss = running_loss / LOG_EVERY
                accuracy = running_correct / running_counted if running_counted > 0 else 0.0
                elapsed = time.time() - epoch_start
                print(f"  Batch {batch_index+1}/{n_batches} | loss {mean_loss:.4f} | move accuracy {accuracy:.4f} | {elapsed:.0f}s elapsed")
                running_loss = 0.0
                running_correct = 0
                running_counted = 0

        print(f"Epoch {epoch+1} training finished in {time.time()-epoch_start:.0f}s. Evaluating...")

        # Measure the solve rate on the held-out cube states. solve() runs on CPU-side CubeState
        # objects, so keep the model wherever it already is and let solve() handle eval mode.
        solve_rate = evaluate_solve_rate(model, eval_cubestates)

        # Save a checkpoint at the end of every epoch
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save({
            "epoch": epoch + 1,
            "solve_rate": solve_rate,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_size": input_size,
            "output_size": output_size,
            "n_blocks": N_BLOCKS,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
        }, CHECKPOINT_PATH)
        print(f"  Saved checkpoint to {CHECKPOINT_PATH}")

    return model


if __name__ == "__main__":
    train()
