import torch
from torch import nn
import torch.nn.functional as F

from Code.Data.CubeState import CubeState, MOVE_SEQUENCE, SOLVED_STATE

device = "cuda" if torch.cuda.is_available() else torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def X_transform(X):
    #return torch.tensor((X-3)/3)
    return X

def Y_transform(y):
    #return torch.tensor(y+20/40)
    return y


# --------------------------------------------------------------------------- #
# Staged sticker masking
# --------------------------------------------------------------------------- #
# Color codes (from CubeState): r=0, y=1, g=2, w=3, o=4, b=5.
WHITE = 3
YELLOW = 1

# The 12 fixed edge slots, each a pair of flat sticker indices (taken from
# CubeState's DEEPCUBE edge sticker map). Reading the two colors currently at a
# slot identifies which edge PIECE occupies it, wherever that piece has moved.
EDGE_SLOTS = [[30, 7], [34, 52], [32, 43], [28, 25],   # white edges (home slots)
              [5, 21], [3, 50], [41, 48], [39, 23],     # middle-layer edges
              [12, 1], [16, 19], [14, 37], [10, 46]]    # yellow edges

# The 8 fixed corner slots, each a triple of flat sticker indices (from the
# DEEPCUBE corner sticker map). Same idea: read the three colors to know which
# corner PIECE currently occupies the slot.
CORNER_SLOTS = [[27, 8, 24], [33, 53, 6], [35, 44, 51], [29, 26, 42],   # white corners
                [9, 0, 47], [15, 18, 2], [17, 36, 20], [11, 45, 38]]    # yellow corners

# The five non-yellow centers (red, green, white, orange, blue). Centers never
# move, so these fixed positions always carry their own color.
NON_YELLOW_CENTER_POSITIONS = [4, 22, 31, 40, 49]

# The 8 stickers of the white cross when solved (the four white edges' home slots).
WHITE_CROSS_POSITIONS = [30, 7, 34, 52, 32, 43, 28, 25]

# The 21 last-layer stickers: yellow edges + yellow corners + yellow center.
LAST_LAYER_POSITIONS = set(
    [12, 1, 16, 19, 14, 37, 10, 46] +                    # yellow edges (8)
    [9, 0, 47, 15, 18, 2, 17, 36, 20, 11, 45, 38] +      # yellow corners (12)
    [13]                                                 # yellow center (1)
)
# Everything that is NOT last layer must be solved for F2L to be complete.
FIRST_TWO_LAYER_POSITIONS = [i for i in range(54) if i not in LAST_LAYER_POSITIONS]


def _cross_complete(state):
    """White cross done: the four white edges sit solved in their home slots."""
    return all(state[i] == SOLVED_STATE[i] for i in WHITE_CROSS_POSITIONS)


def _f2l_complete(state):
    """First two layers done: everything except the last (yellow) layer is solved."""
    return all(state[i] == SOLVED_STATE[i] for i in FIRST_TWO_LAYER_POSITIONS)


class StagedStickerMask:
    """
    Stateful input transform that reveals only the stickers relevant to the
    current solving stage, and hides the rest.

    Output is always a length-54 vector. Hidden stickers are set to 0; visible
    stickers are set to (color_code + 1), so that a hidden 0 is never confused
    with red (color 0) -> after masking, red=1, yellow=2, green=3, white=4,
    orange=5, blue=6.

    Stages (advanced monotonically as the solve progresses):
        0  White cross  -> the four white edges + the five non-yellow centers
                           (8 + 5 = 13 stickers).
        1  First two layers -> every edge with no yellow sticker (white + middle
                               edges) + every corner that has a white sticker +
                               the five non-yellow centers (16 + 12 + 5 = 33).
        2  Last layer   -> show all 54 stickers.

    Edge/corner visibility is piece-based: each fixed slot is inspected for the
    colors it currently holds, so a white edge (or white corner) is revealed no
    matter where on the cube it happens to be.
    """

    STAGE_WHITE_CROSS = 0
    STAGE_F2L = 1
    STAGE_LAST_LAYER = 2

    def __init__(self):
        self.stage = self.STAGE_WHITE_CROSS

    def reset(self):
        """Reset to stage 0. Call at the start of each new solve."""
        self.stage = self.STAGE_WHITE_CROSS
        return self

    def _advance(self, state):
        # Monotonic: only ever move forward through the stages.
        if self.stage == self.STAGE_WHITE_CROSS and _cross_complete(state):
            self.stage = self.STAGE_F2L
        if self.stage == self.STAGE_F2L and _f2l_complete(state):
            self.stage = self.STAGE_LAST_LAYER

    def _visible_positions(self, state):
        if self.stage >= self.STAGE_LAST_LAYER:
            return range(54)  # last layer: everything is visible

        visible = []

        # Edges (piece-based: read the colors currently at each slot).
        for a, b in EDGE_SLOTS:
            colors = (state[a], state[b])
            if self.stage == self.STAGE_WHITE_CROSS:
                show = WHITE in colors                 # only white edges
            else:  # STAGE_F2L
                show = YELLOW not in colors            # every non-yellow edge
            if show:
                visible.append(a)
                visible.append(b)

        # The five non-yellow centers are visible from stage 0 onward.
        visible.extend(NON_YELLOW_CENTER_POSITIONS)

        # From stage 1, also reveal every corner that holds a white sticker.
        if self.stage == self.STAGE_F2L:
            for slot in CORNER_SLOTS:
                if WHITE in (state[slot[0]], state[slot[1]], state[slot[2]]):
                    visible.extend(slot)

        return visible

    def __call__(self, state):
        """
        Mask a single cube state.

        Args:
            state: length-54 iterable of raw color codes (0-5).

        Returns:
            list of 54 ints: 0 where hidden, (color_code + 1) where visible.
        """
        state = [int(c) for c in state]
        self._advance(state)                           # detect stage first
        masked = [0] * 54
        for i in self._visible_positions(state):
            masked[i] = state[i] + 1                   # +1 so hidden(0) != red(0)
        return masked


class SolverRNN(nn.Module):
    """
    Custom LSTM recurrent network that solves a Rubik's cube one move at a time.

    Architecture (per time step t):
        - The cube configuration x_t (a length-54 vector) is the raw input.
        - The previous time step's one-hot move output o_{t-1} (length `output_size`)
          is fed as an ADDITIONAL input into ALL three LSTM hidden layers.
        - Layer 0 cell sees  concat(x_t,  o_{t-1})
          Layer 1 cell sees  concat(h0_t, o_{t-1})
          Layer 2 cell sees  concat(h1_t, o_{t-1})
        - A linear head maps the top layer's hidden state h2_t -> logits over the
          20 possible outputs (`MOVE_SEQUENCE`: 18 turns + "nothing" + "stop").
        - The argmax of those logits is the predicted move; its one-hot becomes
          o_t, the feedback that is injected into all three layers next step.

    Each LSTMCell carries its own (h, c) state across time steps, so the network
    is genuinely recurrent in depth-of-search, not just in feature mixing.
    """

    def __init__(self, input_size=54, hidden_size=1080, hidden_layers=3, output_size=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        # One output per possible move/token in MOVE_SEQUENCE (18 turns + '-' + '#').
        self.output_size = output_size if output_size is not None else len(MOVE_SEQUENCE)

        # Build the stack of LSTM cells. Every cell's input is augmented with the
        # previous step's output (size = self.output_size), so the feature input to
        # each cell is (its own feature size) + output_size.
        self.lstm_cells = nn.ModuleList()
        for layer in range(hidden_layers):
            feature_size = input_size if layer == 0 else hidden_size
            cell_input_size = feature_size + self.output_size
            self.lstm_cells.append(nn.LSTMCell(cell_input_size, hidden_size, bias=True))

        # Maps the top hidden state to logits over the move vocabulary.
        self.output_layer = nn.Linear(hidden_size, self.output_size)

    # ------------------------------------------------------------------ #
    # Core building blocks
    # ------------------------------------------------------------------ #

    def _init_hidden(self, batch_size, dev):
        """Returns a list of (h, c) zero-tuples, one per LSTM layer."""
        hidden = []
        for _ in range(self.hidden_layers):
            h = torch.zeros(batch_size, self.hidden_size, device=dev)
            c = torch.zeros(batch_size, self.hidden_size, device=dev)
            hidden.append((h, c))
        return hidden

    def _step(self, x_t, prev_out, hidden):
        """
        Advance the network by a single time step.

        Args:
            x_t:      (batch, input_size)  current cube configuration
            prev_out: (batch, output_size) previous step's output (one-hot or probs)
            hidden:   list of (h, c) tuples, one per layer

        Returns:
            logits:     (batch, output_size) raw scores for the next move
            new_hidden: updated list of (h, c) tuples
        """
        new_hidden = []
        # Layer 0 consumes the cube state; deeper layers consume the layer below's
        # hidden state. In every case the previous output is concatenated on.
        layer_feature = x_t
        for i, cell in enumerate(self.lstm_cells):
            cell_input = torch.cat([layer_feature, prev_out], dim=-1)
            h, c = cell(cell_input, hidden[i])
            new_hidden.append((h, c))
            layer_feature = h  # feed this layer's output up to the next layer

        logits = self.output_layer(new_hidden[-1][0])
        return logits, new_hidden

    # ------------------------------------------------------------------ #
    # Training pass (teacher forcing over fixed-length sequences)
    # ------------------------------------------------------------------ #

    def train_forward(self, x_seq, target_seq=None):
        """
        Process a batch of cube-state sequences and return per-step logits.

        Args:
            x_seq:      (batch, T, input_size) sequence of cube configurations.
            target_seq: (batch, T) optional ground-truth move indices. When given,
                        their one-hots are fed back as the "previous output"
                        (teacher forcing). When omitted, the network's own
                        softmax output is fed back (differentiable feedback).

        Returns:
            logits_seq: (batch, T, output_size)
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)  # allow a single unbatched sequence

        batch, T, _ = x_seq.shape
        dev = x_seq.device

        hidden = self._init_hidden(batch, dev)
        prev_out = torch.zeros(batch, self.output_size, device=dev)

        logits_seq = []
        for t in range(T):
            logits, hidden = self._step(x_seq[:, t, :], prev_out, hidden)
            logits_seq.append(logits)

            if target_seq is not None:
                # Teacher forcing: feed the true previous move as a one-hot.
                # clamp(min=0) tolerates padded positions (e.g. the -100 ignore
                # index): their feedback only ever flows into other padded steps
                # whose loss is masked out, so the substituted value is harmless.
                prev_out = F.one_hot(target_seq[:, t].clamp(min=0).long(), self.output_size).float()
            else:
                # Differentiable feedback of the network's own prediction.
                prev_out = F.softmax(logits, dim=-1)

        return torch.stack(logits_seq, dim=1)

    # ------------------------------------------------------------------ #
    # Inference pass (autoregressive solve loop)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward(self, x, max_moves=200):
        """
        Solve a cube autoregressively.

        Pass in a single 1D vector of length 54 representing the cube
        configuration. The network repeatedly:
            1. predicts the next-best move (one-hot argmax over MOVE_SEQUENCE),
            2. applies that turn to a CubeState,
            3. feeds the resulting configuration in as the next input while
               passing its last output into all three hidden layers,
        until the cube is solved or `max_moves` (200) is reached.

        Returns:
            moves: a list of move letters (e.g. ["R", "U'", "F2", ...]) — the
                   solution sequence, in order.
        """
        self.eval()

        # Normalize the input to a list of 54 ints for CubeState, and a (1, 54) tensor.
        if isinstance(x, torch.Tensor):
            flat = [int(round(v)) for v in x.detach().flatten().tolist()]
        else:
            flat = [int(round(float(v))) for v in list(x)]

        if len(flat) != self.input_size:
            raise ValueError(f"Expected an input of length {self.input_size}, got {len(flat)}.")

        cube = CubeState(flat)

        # Already solved? Nothing to do.
        if cube.get_flat_data() == SOLVED_STATE:
            return []

        # Run on whatever device the model's parameters currently live on.
        dev = next(self.parameters()).device
        hidden = self._init_hidden(1, dev)
        prev_out = torch.zeros(1, self.output_size, device=dev)

        # Staged mask hides stickers irrelevant to the current solving stage and
        # advances (white cross -> F2L -> last layer) as the cube gets solved.
        mask = StagedStickerMask()

        moves = []
        for _ in range(max_moves):
            masked_state = mask(cube.get_flat_data())
            x_t = torch.tensor(masked_state, dtype=torch.float32, device=dev).unsqueeze(0)

            logits, hidden = self._step(x_t, prev_out, hidden)

            move_idx = int(logits.argmax(dim=-1).item())
            move = MOVE_SEQUENCE[move_idx]

            # The one-hot of this prediction is the feedback for the next step.
            prev_out = F.one_hot(torch.tensor([move_idx], device=dev), self.output_size).float()

            if move == '#':            # explicit stop token
                break
            if move == '-':            # "nothing" — no turn applied this step
                continue

            cube.move(move)
            moves.append(move)

            if cube.get_flat_data() == SOLVED_STATE:
                break

        return moves
