"""
Hard-coded Rubik's Cube solver for the Jetson Nano robot.
Solves: White Cross -> F2L -> Last Layer (OLL + PLL)
Physical orientation: Green=F (front), Red=L (left), White=D (down)
Uses CubeState.flat_data and performs moves via ClawMachine.
Each move is executed on both the ClawMachine (physical) and CubeState.move() (data).

Usage:
    from CubeState import CubeState
    from ClawMachine import ClawMachine
    from HardCodedSolver import HardCodedSolver, solve_cube

    cube = CubeState(initial_state_data)
    claw = ClawMachine()
    solver = HardCodedSolver(cube, claw, simulate=False)
    solver.solve()  # Performs moves on robot and updates cube state

    # Or use the convenience function:
    solve_cube(cube, claw)  # With robot
    solve_cube(cube, None, simulate=True)  # Simulation only (no hardware)
"""

from CubeState import CubeState, SOLVED_STATE

try:
    from ClawMachine import ClawMachine
except (ImportError, ModuleNotFoundError):
    ClawMachine = None  # Optional when simulating without robot hardware

# Color indices: 0=Red(L), 1=Yellow(U), 2=Green(F), 3=White(D), 4=Orange(R), 5=Blue(B)
WHITE = 3
YELLOW = 1
GREEN = 2
RED = 0
ORANGE = 4
BLUE = 5

# Face indices: 0=L, 1=U, 2=F, 3=D, 4=R, 5=B
# Flat index = face*9 + position (0-8, row-major)
def idx(face, pos):
    return face * 9 + pos

# D-face edge positions for white cross: (d_sticker_idx, adj_sticker_idx), adj_center_color
# From CubeExtractor: F-D=(2,7)-(3,1), L-D=(0,7)-(3,3), R-D=(4,7)-(3,5), B-D=(5,7)-(3,7)
WHITE_CROSS_TARGETS = [
    (idx(3, 1), idx(2, 7), GREEN),   # D[1]-F[7], white-green
    (idx(3, 3), idx(0, 7), RED),     # D[3]-L[7], white-red
    (idx(3, 5), idx(4, 7), ORANGE),  # D[5]-R[7], white-orange
    (idx(3, 7), idx(5, 7), BLUE),    # D[7]-B[7], white-blue
]

# All edge piece positions (face, pos), (face, pos)
EDGES = [
    [(0, 1), (1, 3)],   # L-U
    [(1, 7), (2, 1)],   # U-F
    [(1, 5), (4, 1)],   # U-R
    [(1, 1), (5, 1)],   # U-B
    [(0, 5), (2, 3)],   # L-F
    [(2, 5), (4, 3)],   # F-R
    [(4, 5), (5, 3)],   # R-B
    [(5, 5), (0, 3)],   # B-L
    [(0, 7), (3, 3)],   # L-D
    [(2, 7), (3, 1)],   # F-D
    [(4, 7), (3, 5)],   # R-D
    [(5, 7), (3, 7)],   # B-D
]

# Corner positions: (face, pos), (face, pos), (face, pos)
CORNERS = [
    [(1, 6), (2, 0), (0, 2)],   # U-F-L
    [(1, 8), (4, 0), (2, 2)],   # U-R-F
    [(5, 0), (4, 2), (1, 2)],   # B-R-U
    [(1, 0), (0, 0), (5, 2)],   # U-L-B
    [(3, 0), (0, 8), (2, 6)],   # D-L-F
    [(3, 2), (2, 8), (4, 6)],   # D-F-R
    [(3, 8), (4, 8), (5, 6)],   # D-R-B
    [(3, 6), (5, 8), (0, 6)],   # D-B-L
]


class HardCodedSolver:
    def __init__(self, cube_state: CubeState, claw_machine: ClawMachine, simulate=False):
        """
        Args:
            cube_state: The current cube state to solve
            claw_machine: Robot to perform physical moves (or None if simulate=True)
            simulate: If True, only update CubeState, don't call ClawMachine
        """
        self.cube = cube_state
        self.claw = claw_machine
        self.simulate = simulate
        self.move_count = 0

    def execute_move(self, move: str):
        """Perform a move: update cube state and optionally the physical robot."""
        if not self.simulate and self.claw:
            self.claw.move(move)
        self.cube.move(move)
        self.move_count += 1

    def execute_moves(self, moves: list):
        """Execute a sequence of moves."""
        for m in moves:
            self.execute_move(m)

    def solve(self):
        """Full solve: white cross -> F2L -> last layer."""
        self._solve_white_cross()
        if self._is_solved():
            return
        self._solve_f2l()
        if self._is_solved():
            return
        self._solve_last_layer()

    def _solve_white_cross(self):
        """Daisy then insert: (1) Get all 4 white edges to U, (2) Insert each with face2."""
        for _ in range(15):
            self._daisy_phase()
            self._insert_phase()
            if all(self.cube.flat_data[td] == WHITE and self.cube.flat_data[ta] == ac
                   for td, ta, ac in WHITE_CROSS_TARGETS):
                return
            # Eject flipped edges from D back to U
            for target_d_idx, target_adj_idx, adj_color in WHITE_CROSS_TARGETS:
                if self.cube.flat_data[target_adj_idx] == WHITE and self.cube.flat_data[target_d_idx] == adj_color:
                    m = {idx(2, 7): "F", idx(0, 7): "L", idx(4, 7): "R", idx(5, 7): "B"}[target_adj_idx]
                    self.execute_move(m)
                    break

    def _daisy_phase(self):
        """Get all 4 white edges onto U. Only use face moves that don't corrupt solved D slots."""
        face_map = {0: "L", 2: "F", 4: "R", 5: "B"}
        adj_to_face = {idx(2, 7): 2, idx(0, 7): 0, idx(4, 7): 4, idx(5, 7): 5}
        def slot_ok(adj_idx):
            for td, ta, ac in WHITE_CROSS_TARGETS:
                if ta == adj_idx:
                    return self.cube.flat_data[td] == WHITE and self.cube.flat_data[ta] == ac
            return False
        u_edges = [(1, 3), (1, 7), (1, 5), (1, 1)]
        u_partner = {(1, 3): (0, 1), (1, 7): (2, 1), (1, 5): (4, 1), (1, 1): (5, 1)}
        for _ in range(24):
            white_on_u = sum(1 for ue in u_edges for p in [u_partner[ue]]
                            if WHITE in (self.cube.flat_data[idx(ue[0], ue[1])], self.cube.flat_data[idx(p[0], p[1])]))
            if white_on_u >= 4:
                return
            for edge in EDGES[4:8]:
                i1, i2 = idx(edge[0][0], edge[0][1]), idx(edge[1][0], edge[1][1])
                c1, c2 = self.cube.flat_data[i1], self.cube.flat_data[i2]
                if WHITE not in (c1, c2):
                    continue
                f1, f2 = edge[0][0], edge[1][0]
                key = (min(f1, f2), max(f1, f2))
                middle_to_adj = {(0,2): (idx(0,7), idx(2,7)), (2,4): (idx(2,7), idx(4,7)),
                                 (4,5): (idx(4,7), idx(5,7)), (0,5): (idx(0,7), idx(5,7))}
                adj1, adj2 = middle_to_adj.get(key, (idx(2,7), idx(4,7)))
                # Face that brings edge to U (not D): R-B needs B, others either work
                brings_to_U = {(0,2): (0, 2), (2,4): (2, 4), (4,5): (5,), (0,5): (0, 5)}  # R-B: only B
                good_faces = brings_to_U.get(key, (f1, f2))
                order = [f for f in (f1, f2) if f in good_faces] or [f1, f2]
                if len(order) == 2 and c1 != WHITE:
                    order = [f2, f1]  # Prefer face with white for correct orientation
                for face in order:
                    adj = idx(face, 7)
                    if slot_ok(adj):
                        continue  # Don't corrupt this slot
                    u_pos = {0: (1, 3), 2: (1, 7), 4: (1, 5), 5: (1, 1)}[face]
                    ui, pi = idx(u_pos[0], u_pos[1]), idx(u_partner[u_pos][0], u_partner[u_pos][1])
                    if WHITE not in (self.cube.flat_data[ui], self.cube.flat_data[pi]):
                        self.execute_move(face_map[face])
                        break
                else:
                    for face in order:
                        if not slot_ok(idx(face, 7)):
                            self.execute_move(face_map[face])
                            break
                break
            else:
                for edge in EDGES[8:12]:
                    i1, i2 = idx(edge[0][0], edge[0][1]), idx(edge[1][0], edge[1][1])
                    if WHITE not in (self.cube.flat_data[i1], self.cube.flat_data[i2]):
                        continue
                    f = edge[0][0] if edge[1][0] == 3 else edge[1][0]
                    if not slot_ok(idx(f, 7)):
                        self.execute_move(face_map[f])
                    break

    def _insert_phase(self):
        """Insert each white edge from U to D. If flipped, send to middle for daisy to re-fetch."""
        face_map = {0: "L", 2: "F", 4: "R", 5: "B"}
        check = {2: (idx(1, 7), idx(2, 1)), 0: (idx(1, 3), idx(0, 1)),
                 4: (idx(1, 5), idx(4, 1)), 5: (idx(1, 1), idx(5, 1))}
        to_middle = {idx(2, 7): ["U2", "B"], idx(0, 7): ["U2", "R"],
                     idx(4, 7): ["U2", "L"], idx(5, 7): ["U2", "F"]}
        for target_d_idx, target_adj_idx, adj_color in WHITE_CROSS_TARGETS:
            if self.cube.flat_data[target_d_idx] == WHITE and self.cube.flat_data[target_adj_idx] == adj_color:
                continue
            target_face = target_adj_idx // 9
            check_idx, check_adj = check[target_face]
            for _ in range(4):
                data = self.cube.flat_data
                if WHITE in (data[check_idx], data[check_adj]) and adj_color in (data[check_idx], data[check_adj]):
                    if data[check_idx] == WHITE and data[check_adj] == adj_color:
                        self.execute_move(face_map[target_face] + "2")
                        break
                    else:
                        # Flipped: send to middle so next daisy fetches with correct orientation
                        self.execute_moves(to_middle[target_adj_idx])
                        break
                self.execute_move("U")

    def _insert_white_edge(self, target_d_idx, target_adj_idx, adj_color, _depth=0):
        """Insert one white edge into its correct D-face position."""
        if _depth > 10:  # Prevent infinite recursion
            return
        data = self.cube.flat_data
        if data[target_d_idx] == WHITE and data[target_adj_idx] == adj_color:
            return  # Already solved

        # Find the white edge containing (white, adj_color)
        for edge in EDGES:
            i1, i2 = idx(edge[0][0], edge[0][1]), idx(edge[1][0], edge[1][1])
            c1, c2 = data[i1], data[i2]
            if WHITE in (c1, c2) and adj_color in (c1, c2):
                self._bring_white_edge_to_slot(edge, (i1, i2), (c1, c2),
                                               target_d_idx, target_adj_idx, adj_color, _depth)
                return

    def _bring_white_edge_to_slot(self, edge, indices, colors, target_d, target_adj, adj_color, _depth=0):
        """Bring a white edge from its current position to the target D slot."""
        i1, i2 = indices
        c1, c2 = colors
        white_on_first = (c1 == WHITE)

        # Map edge positions to location categories
        face1, pos1 = edge[0]
        face2, pos2 = edge[1]

        # If edge is in target position but flipped, do F2 or similar to flip
        if (i1, i2) == (target_d, target_adj) or (i2, i1) == (target_d, target_adj):
            if self.cube.flat_data[target_d] != WHITE:
                self._flip_edge_in_slot(target_adj, _depth)  # Edge is upside down
            return

        # Edge in middle layer (faces 0,2,4,5, not 1 or 3)
        if face1 != 1 and face2 != 1 and face1 != 3 and face2 != 3:
            self._insert_edge_from_middle(edge, white_on_first, target_adj, _depth)
            return

        # Edge in top layer (U face)
        if 1 in (face1, face2):
            self._insert_edge_from_top(edge, white_on_first, target_adj, adj_color)
            return

        # Edge in bottom layer (D face) but wrong slot
        self._insert_edge_from_bottom(edge, white_on_first, target_adj, target_d, adj_color, _depth)

    def _flip_edge_in_slot(self, adj_idx, _depth=0):
        """Flip an edge in the correct slot but inverted - take out and re-insert."""
        adj_to_target = {idx(2, 7): (idx(3, 1), idx(2, 7), GREEN),
                        idx(0, 7): (idx(3, 3), idx(0, 7), RED),
                        idx(4, 7): (idx(3, 5), idx(4, 7), ORANGE),
                        idx(5, 7): (idx(3, 7), idx(5, 7), BLUE)}
        face_move = {idx(2, 7): "F", idx(0, 7): "L", idx(4, 7): "R", idx(5, 7): "B"}
        self.execute_move(face_move[adj_idx])  # Takes edge to middle (or U)
        target_d, target_adj, adj_color = adj_to_target[adj_idx]
        self._insert_white_edge(target_d, target_adj, adj_color, _depth + 1)

    def _insert_edge_from_middle(self, edge, white_on_first, target_adj, _depth=0):
        """Insert white edge from middle layer to D.
        Use the face whose D-slot is NOT yet solved, to avoid corrupting placed edges."""
        face1, pos1 = edge[0]
        face2, pos2 = edge[1]
        face_map = {0: "L", 2: "F", 4: "R", 5: "B"}
        def slot_solved(adj_idx):
            for td, ta, ac in WHITE_CROSS_TARGETS:
                if ta == adj_idx:
                    return (self.cube.flat_data[td] == WHITE and
                            self.cube.flat_data[ta] == ac)
            return False
        key = (min(face1, face2), max(face1, face2))
        middle_to_adj = {(0, 2): (idx(0, 7), idx(2, 7)), (2, 4): (idx(2, 7), idx(4, 7)),
                        (4, 5): (idx(4, 7), idx(5, 7)), (0, 5): (idx(0, 7), idx(5, 7))}
        adj1, adj2 = middle_to_adj.get(key, (idx(2, 7), idx(4, 7)))
        # Use face whose slot is NOT solved (avoid corrupting placed edges)
        face1_solved = slot_solved(adj1)
        face2_solved = slot_solved(adj2)
        face_to_adj = {0: idx(0, 7), 2: idx(2, 7), 4: idx(4, 7), 5: idx(5, 7)}
        f1, f2 = (face1, face2) if face1 < face2 else (face2, face1)
        if not face1_solved and face2_solved:
            use_face = f1
        elif face1_solved and not face2_solved:
            use_face = f2
        else:
            # Prefer target face if it's one of them
            target_face = target_adj // 9
            use_face = f1 if f1 == target_face else f2
        move = face_map[use_face]
        self.execute_move(move)  # Single face move brings middle edge to U
        for target_d_idx, target_adj_idx, adj_color in WHITE_CROSS_TARGETS:
            if target_adj_idx == target_adj:
                self._insert_white_edge(target_d_idx, target_adj_idx, adj_color, _depth + 1)
                return

    def _insert_edge_from_top(self, edge, white_on_first, target_adj, adj_color):
        """Insert white edge from U layer. Only insert when white is on U (correct orientation).
        If flipped, move to middle (face2 U2 face) then re-insert."""
        target_face = {idx(2, 7): 2, idx(0, 7): 0, idx(4, 7): 4, idx(5, 7): 5}[target_adj]
        face_map = {0: "L", 2: "F", 4: "R", 5: "B"}
        target_move = face_map[target_face]
        check = {2: (idx(1, 7), idx(2, 1)), 0: (idx(1, 3), idx(0, 1)),
                 4: (idx(1, 5), idx(4, 1)), 5: (idx(1, 1), idx(5, 1))}
        check_idx, check_adj = check[target_face]
        for _ in range(4):
            data = self.cube.flat_data
            if WHITE in (data[check_idx], data[check_adj]) and adj_color in (data[check_idx], data[check_adj]):
                if data[check_idx] == WHITE and data[check_adj] == adj_color:
                    self.execute_move(target_move + "2")
                    return
                # else: flipped - skip (rotate U); flipped edges handled by _flip_edge_in_slot
            self.execute_move("U")

    def _insert_edge_from_bottom(self, edge, white_on_first, target_adj, target_d, adj_color, _depth=0):
        """Edge is on D but in wrong slot. Rotate U so we swap with yellow edge, then face2."""
        face1, pos1 = edge[0]
        face2, pos2 = edge[1]
        curr_adj = idx(face1, pos1) if face2 == 3 else idx(face2, pos2)
        d_adj_map = {idx(2, 7): "F", idx(0, 7): "L", idx(4, 7): "R", idx(5, 7): "B"}
        u_above_d = {idx(2, 7): (idx(1, 7), idx(2, 1)), idx(0, 7): (idx(1, 3), idx(0, 1)),
                     idx(4, 7): (idx(1, 5), idx(4, 1)), idx(5, 7): (idx(1, 1), idx(5, 1))}
        face_move = d_adj_map[curr_adj]
        for _ in range(4):
            u_idxs = u_above_d.get(curr_adj, (idx(1, 7), idx(2, 1)))
            u_colors = {self.cube.flat_data[u_idxs[0]], self.cube.flat_data[u_idxs[1]]}
            if YELLOW in u_colors:
                break
            self.execute_move("U")
        self.execute_move(face_move + "2")
        self._insert_white_edge(target_d, target_adj, adj_color, _depth + 1)

    def _solve_f2l(self):
        """Solve F2L - insert the 4 corners (edges already in cross)."""
        slot_colors = [
            (WHITE, GREEN, RED),     # F-L slot
            (WHITE, GREEN, ORANGE),  # F-R slot
            (WHITE, BLUE, ORANGE),   # B-R slot
            (WHITE, BLUE, RED),      # B-L slot
        ]
        for corner_colors in slot_colors:
            self._insert_f2l_corner(corner_colors)

    def _insert_f2l_corner(self, target_colors):
        """Insert one corner using beginner method: position above slot, R U R' U'."""
        w, c1, c2 = target_colors
        slot = self._slot_for_corner(c1, c2)
        for _ in range(30):  # Max iterations
            if self._f2l_corner_solved(target_colors):
                return
            data = self.cube.flat_data
            # Find where our corner is
            corner_pos = None
            for corner in CORNERS:
                idxs = [idx(f, p) for f, p in corner]
                cols = [data[i] for i in idxs]
                if set(cols) == {w, c1, c2}:
                    corner_pos = corner
                    break
            if not corner_pos:
                return
            faces_in = {c[0] for c in corner_pos}
            if 3 in faces_in:
                # Corner in bottom - if wrong slot, take it out
                if not self._corner_in_correct_slot(corner_pos, target_colors):
                    if slot == "R":
                        self.execute_moves(["R", "U", "R'", "U'"])
                    else:
                        self.execute_moves(["L'", "U'", "L", "U"])
                else:
                    return  # Already in correct slot
            else:
                # Corner in top - rotate U until above slot, then insert
                for _ in range(4):
                    if self._f2l_corner_above_slot(corner_pos, target_colors, slot):
                        if slot == "R":
                            self.execute_moves(["R", "U", "R'", "U'"])
                        else:
                            self.execute_moves(["L'", "U'", "L", "U"])
                        break
                    self.execute_move("U")

    def _slot_for_corner(self, c1, c2):
        if GREEN in (c1, c2) and RED in (c1, c2):
            return "L"
        if GREEN in (c1, c2) and ORANGE in (c1, c2):
            return "R"
        if BLUE in (c1, c2) and ORANGE in (c1, c2):
            return "R"
        return "L"

    def _f2l_corner_solved(self, target_colors):
        """Check if corner is in correct slot with white on D."""
        w, c1, c2 = target_colors
        # Target slots and their D corner positions
        slot_corners = {
            (GREEN, RED): (3, 0), (GREEN, ORANGE): (3, 2),
            (BLUE, ORANGE): (3, 8), (BLUE, RED): (3, 6),
        }
        key = (c1, c2) if (c1, c2) in slot_corners else (c2, c1)
        dp = slot_corners[key][1]
        adj = self._corner_adjacent_faces(3, dp)
        idxs = [idx(3, dp)] + [idx(f, p) for f, p in adj]
        cols = [self.cube.flat_data[i] for i in idxs]
        if set(cols) != {w, c1, c2}:
            return False
        return self.cube.flat_data[idx(3, dp)] == w  # White must be on D

    def _corner_in_correct_slot(self, corner_pos, target_colors):
        """Check if corner is in its target slot (any orientation)."""
        return self._f2l_corner_solved(target_colors)

    def _corner_adjacent_faces(self, df, dp):
        """Get (face, pos) for the 2 faces adjacent to D corner at dp."""
        d_corner_map = {0: [(0, 8), (2, 6)], 2: [(2, 8), (4, 6)],
                        8: [(4, 8), (5, 6)], 6: [(5, 8), (0, 6)]}
        return d_corner_map.get(dp, [])

    def _f2l_corner_above_slot(self, corner, target_colors, slot):
        """Check if target corner is in a top-layer position above its slot."""
        w, c1, c2 = target_colors
        data = self.cube.flat_data
        corner_idxs = [idx(f, p) for f, p in corner]
        corner_cols = [data[i] for i in corner_idxs]
        if set(corner_cols) != {w, c1, c2}:
            return False
        # Above R slots: U-R-F (1,8),(4,0),(2,2) or U-B-R (1,2),(5,0),(4,2)
        # Above L slots: U-F-L (1,6),(2,0),(0,2) or U-L-B (1,0),(0,0),(5,2)
        above_R = [frozenset([(1, 8), (4, 0), (2, 2)]), frozenset([(1, 2), (5, 0), (4, 2)])]
        above_L = [frozenset([(1, 6), (2, 0), (0, 2)]), frozenset([(1, 0), (0, 0), (5, 2)])]
        corner_set = frozenset((c[0], c[1]) for c in corner)
        targets = above_R if slot == "R" else above_L
        return corner_set in targets

    def _solve_last_layer(self):
        """OLL + PLL for last layer."""
        self._oll()
        self._pll()

    def _oll(self):
        """Orient Last Layer. Tries Sune, Antisune, and F R U R' U' F'."""
        sune = ["R", "U", "R'", "U", "R", "U2", "R'"]
        antisune = ["R'", "U'", "R", "U'", "R'", "U2", "R"]
        frurf = ["F", "R", "U", "R'", "U'", "F'"]
        algs = [sune, antisune, frurf]
        for _ in range(18):  # Try each alg up to 6 times
            if all(self.cube.flat_data[idx(1, i)] == YELLOW for i in range(9)):
                return
            self.execute_moves(algs[_ % 3])

    def _pll(self):
        """Permute Last Layer - put pieces in correct positions."""
        max_iter = 4
        for _ in range(max_iter):
            if self._pll_corners_done() and self._pll_edges_done():
                return
            # Ua perm: R U' R U R U R U' R' U' R2
            self.execute_moves(["R", "U'", "R", "U", "R", "U", "R", "U'", "R'", "U'", "R2"])
            if self._pll_corners_done() and self._pll_edges_done():
                return
            self.execute_move("U")

    def _pll_corners_done(self):
        """Check if U-layer corners are correctly placed (yellow on U, sides match)."""
        data = self.cube.flat_data
        u_corner_checks = [
            (idx(1, 6), idx(2, 0), idx(0, 2), GREEN, RED),    # U-F-L
            (idx(1, 8), idx(4, 0), idx(2, 2), ORANGE, GREEN), # U-R-F
            (idx(1, 0), idx(0, 0), idx(5, 2), RED, BLUE),     # U-L-B
            (idx(1, 2), idx(5, 0), idx(4, 2), BLUE, ORANGE),  # U-B-R
        ]
        for uc, a, b, c1, c2 in u_corner_checks:
            if data[uc] != YELLOW:
                return False
            if {data[a], data[b]} != {c1, c2}:
                return False
        return True

    def _pll_edges_done(self):
        """Check if U-layer edges match their adjacent face centers."""
        data = self.cube.flat_data
        # U edges: U[1]-B, U[3]-L, U[5]-R, U[7]-F
        u_edge_checks = [(idx(1, 1), idx(5, 1)), (idx(1, 3), idx(0, 1)),
                        (idx(1, 5), idx(4, 1)), (idx(1, 7), idx(2, 1))]
        for ue, adj in u_edge_checks:
            face = adj // 9
            if data[adj] != data[idx(face, 4)]:
                return False
        return True

    def _is_solved(self):
        """Check if the entire cube is solved."""
        return self.cube.flat_data == SOLVED_STATE


def solve_cube(cube_state: CubeState, claw_machine=None, simulate: bool = False):
    """
    Solve the cube using the hard-coded algorithm.
    If claw_machine is None and simulate is False, only cube state is updated (no physical moves).
    """
    if claw_machine is None and not simulate:
        simulate = True
    solver = HardCodedSolver(cube_state, claw_machine, simulate=simulate)
    solver.solve()
    return solver.move_count


if __name__ == "__main__":
    # Test with simulation (no robot)
    cube = CubeState()
    scramble = ["R", "U", "L'", "D'", "R'", "D", "B", "D", "L", "U", "F2"]
    for m in scramble:
        cube.move(m)
    print("Scrambled:", cube.flat_data)
    solver = HardCodedSolver(cube, None, simulate=True)
    solver.solve()
    print("Move count:", solver.move_count)
    print("Solved:", solver._is_solved())
    print(cube.flat_data)
