from sys import hash_info

from Camera import Camera  # UNCOMMENT THIS
from PIL import Image
from CubeState import COLOR_SEQUENCE, FACE_SEQUENCE, CubeState, flatten_data
import time, cv2, random
import numpy as np
from VisionRunner import Model # UNCOMMENT THIS
from VisionPiece import Piece
from RubikTestProbabilities import TestProbabilities

LEGAL_PIECES = {
    "corners" : [[3, 0, 2], [3, 5, 0], [3, 4, 5], [3, 2, 4], [1, 0, 5], [1, 2, 0], [1, 4, 2], [1, 5, 4]],
    "edges" : [[3, 0], [3, 5], [3, 4], [3, 2], [0, 2], [0, 5], [4, 5], [4, 2], [1, 0], [1, 2], [1, 4], [1, 5]],
    "centers" : [0, 1, 2, 3, 4, 5]
}



class CubeReader:

    def __init__(self, claw_machine):
        self.claw_machine = claw_machine
        self.model = Model()

    # Procedure:
    # Simply have camera read every face (order doesn't matter) using computer vision.
    # Then, take the list of unordered faces data and order them according to Face Order (L, U, F, ...)
    # Finally, feed this into CubeState, which will be able to flatten it and perform "data cube turns."
    def ReadCube(self, record_data=False, optional_index=0):
        # Returns: CubeState
        print("Reading Cube")
        cam = Camera()
        faces = []
        color_probs = [] # 6x9x6 nested list
        for i in range(7):
            self.claw_machine.face_to_cam(i)
            time.sleep(0.5)
            if i < 6:
                cropped_img = cam.get_cropped_img(i)

                if record_data:
                    #img_to_write = cropped_img.astype(np.uint8, copy=True)
                    #img_to_write = cv2.cvtColor(img_to_write, cv2.COLOR_HSV2BGR)
                    cv2.imwrite(f"src/UncroppedScans/img_{'0'*(4-len(str(optional_index)))}{optional_index}-{str(i)}.jpg", cropped_img)

                # Run model, and store the color probabilities for each of the 9 stickers in the image.
                color_probs.append(self.model.predict_probs(cropped_img))

            time.sleep(0.5)

        # Look at the color probability sets of all 6 images, and make appropriate corrections.
        # Pseudo Code:
        # colors = get_colors_from_probs(color_probs)  # 6x9 nested list
        # Until No Change Happens:
        #   color_corrections = get_col_cor(colors)  # list of len 6, eg: (1, -2, 0, 2, -1, 0) means state needs 1 more red, 2 fewer yellow, etc.
        #   colors = correct_by_probabilities(colors, color_corrections, color_probs)  # For each item in color_corrections, ... (need create logic! both look at color to correct and all other colors from which to possibly take)

        print("Reconstructing cube...")
        colors = self.get_cols_from_probs(color_probs) # A 6x9 list
        #col_corrections = self.get_col_corrections(colors) # A len 6 list. Its elements should add up to 0.
        colors = self.correct_colors(colors, color_probs)

        return colors

    def TestReadCube(self, test_probabilities):
        

        print("Reconstructing cube...")
        colors = self.get_cols_from_probs(test_probabilities) # A 6x9 list
        #col_corrections = self.get_col_corrections(colors) # A len 6 list. Its elements should add up to 0.
        colors = self.correct_colors(colors, test_probabilities)

        return colors

    def correct_colors(self, colors, color_probs):
        # See description of algorithm below [in correct_one_color()]
        corners, edges, centers = self.extract_pieces(colors, color_probs)

        # Match extracted pieces to legal pieces
        print("Matching corners to legal pieces...")
        corner_matches = self.match_pieces_to_legal(corners, LEGAL_PIECES["corners"], "corner")
        
        print("Matching edges to legal pieces...")
        edge_matches = self.match_pieces_to_legal(edges, LEGAL_PIECES["edges"], "edge")
        
        print("Matching centers to legal pieces...")
        center_matches = self.match_pieces_to_legal(centers, LEGAL_PIECES["centers"], "center")
        
        # Update colors based on matches
        print("Updating colors based on matches...")
        colors = self.update_colors_from_matches(colors, corners, edges, centers, corner_matches, edge_matches, center_matches)
        
        return colors

    def correct_one_color(self, color, colors, color_probs, col_corrections):
        if col_corrections[color] <= 0:
            print("WARNING: correct_one_color cannot perform corrections on negatives! Only run it on positives. Negatives are taken care of automatically.")
            return

        # (random idea: for encoding a cubestate, use 3d coordinates for piece position, and orientation (1-3) value)

        # NEW ALGORITHM: reconstruct plausible pieces of the cube by seeing which illegal piece has greatest probability of being each legal piece
        # To identify illegal pieces, first identify the colors on each piece in its standard orientation:
        #   Corner pieces: standard orientation is one sticker Up, one sticker Left, and one sticker Front; White/Yellow on Top. Order: U, F, L. (if no White/Yellow, no rotation; it's illegal!)
        #   (imagine taking a corner piece and moving it to the Upper layer [with cube in Default Orientation], then twisting upper face so corner is in front-left position.)
        #   Edge pieces: standard orientation is one sticker Up, one sticker Front; White/Yellow on Top; or Red/Orange on top. Order: U, F. (imagine edge in top layer, facing front face.)
        #   Center pieces: no need for standard orientation (only one sticker)
        # Create a "parsed cube state" consisting of pieces (which are ordered lists of sticker colors in standard orientation ^^), rather than a list of 54 stickers.
        # Then, compare these pieces to a dictionary of "existing/legal" pieces.
        # If a piece is illegal (no legal matches):
        #   look at each legal piece of this type (corner, edge, etc).
        #   for each legal piece, calculate a value corresponding to how "close" this illegal piece is to each legal piece.
        #   pick the piece that is closest to the illegal piece (changing the sticker with the least confidence, and using alternative stickers with the highest possible probability), MAKING SURE NOT to pick a legal piece that's already present on the cube.

        # OLD ALGORITHM: looks at all stickers of the given color, and finds the best sticker to switch and the best color to switch it to, and performs the switch.
        # You can only switch it to a color that is lacking (as seen in col_corrections).
        # The sticker of 'color' to be switched is the one whose next-best color probability is one of the lacking colors.
        # If multiple such stickers exist, perform the switch on the one with the best priority ranking:
        #   1) The sticker's next-best color is its SIMILAR color (red <-> orange, yellow <-> white, etc.)
        #   2) The sticker's next-best color probability is the highest

        # Can use this loop to check which stickers can be switched for example.
        # But organize code into helper methods, including this loop.

    def extract_pieces(self, colors, probabilities):
        extraction_indices = {
            "corners" : [[[0, 2], [1, 6], [2, 0]],
                         [[1, 8], [4, 0], [2, 2]],
                         [[1, 2], [5, 0], [4, 2]],
                         [[1, 0], [0, 0], [5, 2]],
                         [[2, 6], [3, 0], [0, 8]],
                         [[2, 8], [4, 6], [3, 2]],
                         [[4, 8], [5, 6], [3, 8]],
                         [[0, 6], [3, 6], [5, 8]]],
            "edges" : [[[0, 1], [1, 3]],
                       [[1, 7], [2, 1]],
                       [[1, 5], [4, 1]],
                       [[1, 1], [5, 1]],
                       [[0, 5], [2, 3]],
                       [[2, 5], [4, 3]],
                       [[4, 5], [5, 3]],
                       [[5, 5], [0, 3]],
                       [[0, 7], [3, 3]],
                       [[2, 7], [3, 1]],
                       [[4, 7], [3, 5]],
                       [[5, 7], [3, 7]]],
            "centers" : [[0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4]]
        }

        print("Extracting corners...")
        corners = []
        for corner in extraction_indices["corners"]:
            corner_colors = []
            corner_indices = corner.copy()
            corner_probabilities = []
            for c in corner:
                corner_colors.append(colors[c[0]][c[1]])
                corner_probabilities.append(probabilities[c[0]][c[1]].copy())
            corners.append(Piece(corner_colors, corner_indices, corner_probabilities))

        print("Extracting edges...")
        edges = []
        for edge in extraction_indices["edges"]:
            edge_colors = []
            edge_indices = edge.copy()
            edge_probabilities = []
            for e in edge:
                edge_colors.append(colors[e[0]][e[1]])
                edge_probabilities.append(probabilities[e[0]][e[1]].copy())
            edges.append(Piece(edge_colors, edge_indices, edge_probabilities))

        print("Extracting centers...")
        centers = []
        for center in extraction_indices["centers"]:
            center_colors = [colors[center[0]][center[1]]]
            center_indices = [center.copy()]
            center_probabilities = [probabilities[center[0]][center[1]]]
            centers.append(Piece(center_colors, center_indices, center_probabilities))


        return corners, edges, centers

    def match_pieces_to_legal(self, extracted_pieces, legal_pieces, piece_type):
        """
        Matches extracted pieces to legal pieces using a greedy matching algorithm.
        Returns a list of tuples: (extracted_piece_index, legal_piece_index, rotation)
        """
        n = len(extracted_pieces)
        matches = []
        used_legal_indices = set()
        
        # Calculate match scores for all pairs
        match_scores = []
        for i, extracted in enumerate(extracted_pieces):
            scores_for_extracted = []
            for j, legal in enumerate(legal_pieces):
                if piece_type == "corner":
                    score, rotation = self.calculate_corner_match_score(extracted, legal)
                elif piece_type == "edge":
                    score, rotation = self.calculate_edge_match_score(extracted, legal)
                else:  # center
                    score, rotation = self.calculate_center_match_score(extracted, legal)
                scores_for_extracted.append((score, j, rotation))
            match_scores.append(scores_for_extracted)
        
        # Greedy matching: sort all possible matches by score and assign best matches
        all_matches = []
        for i in range(n):
            for score, j, rotation in match_scores[i]:
                all_matches.append((score, i, j, rotation))
        
        # Sort by score (lower is better, as it represents cost/distance)
        all_matches.sort(key=lambda x: x[0])
        
        # Assign matches greedily, but ensuring center pieces follow legal opposite configuration
        OPPOSITES = [4, 3, 5, 1, 0, 2]
        used_extracted = set()
        for score, i, j, rotation in all_matches:
            if i not in used_extracted and j not in used_legal_indices:
                matches.append((i, j, rotation))
                used_extracted.add(i)
                used_legal_indices.add(j)
                if piece_type == "center":
                    matches.append((OPPOSITES[i], OPPOSITES[j], rotation))
                    used_extracted.add(OPPOSITES[i])
                    used_legal_indices.add(OPPOSITES[j])
                
        # Validate that all pieces were matched
        if len(matches) != n:
            print(f"WARNING: Only matched {len(matches)} out of {n} {piece_type} pieces!")
        if len(used_legal_indices) != len(legal_pieces):
            print(f"WARNING: Only used {len(used_legal_indices)} out of {len(legal_pieces)} legal {piece_type} pieces!")
        
        return matches

    def calculate_corner_match_score(self, extracted_piece, legal_piece):
        """
        Calculates how well an extracted corner piece matches a legal corner piece.
        Returns (score, rotation) where rotation is the best rotation (0-2).
        Lower score is better.
        Uses logits: higher logit = more confident prediction.
        """
        best_score = float('inf')
        best_rotation = 0
        
        # Try all 3 rotations
        for rotation in range(3):
            score = 0
            # Compare each sticker
            for sticker_idx in range(3):
                extracted_color = extracted_piece.colors[sticker_idx]
                legal_color = legal_piece[(sticker_idx + rotation) % 3] # TODO: This rotates the piece anticlockwise, not clockwise
                
                # Get logit for the legal color at this sticker position
                logit = extracted_piece.probabilities[sticker_idx][legal_color]
                
                if extracted_color == legal_color:
                    # Reward: subtract logit (higher logit = more confident = better match)
                    # Since logits can be negative, we want to subtract to make score lower
                    score -= logit
                else:
                    # Penalty: add a cost based on how low the logit is
                    # Lower logit means less confidence, so higher cost
                    # Use negative logit as penalty (if logit is -500, penalty is 500)
                    score += (-logit + 1000)  # Add base cost plus negative logit
            
            if score < best_score:
                best_score = score
                best_rotation = rotation
        
        return best_score, best_rotation

    def calculate_edge_match_score(self, extracted_piece, legal_piece):
        """
        Calculates how well an extracted edge piece matches a legal edge piece.
        Returns (score, rotation) where rotation is 0 or 1 (flipped).
        Lower score is better.
        Uses logits: higher logit = more confident prediction.
        """
        best_score = float('inf')
        best_rotation = 0
        
        # Try both orientations (normal and flipped)
        for rotation in range(2):
            score = 0
            for sticker_idx in range(2):
                extracted_color = extracted_piece.colors[sticker_idx]
                legal_color = legal_piece[(sticker_idx + rotation) % 2]
                
                logit = extracted_piece.probabilities[sticker_idx][legal_color]
                
                if extracted_color == legal_color:
                    score -= logit
                else:
                    score += (-logit + 1000)
            
            if score < best_score:
                best_score = score
                best_rotation = rotation
        
        return best_score, best_rotation

    def calculate_center_match_score(self, extracted_piece, legal_piece):
        """
        Calculates how well an extracted center piece matches a legal center piece.
        Returns (score, 0) since centers don't rotate.
        Lower score is better.
        Uses logits: higher logit = more confident prediction.
        """
        extracted_color = extracted_piece.colors[0]
        legal_color = legal_piece
        
        logit = extracted_piece.probabilities[0][legal_color]
        
        if extracted_color == legal_color:
            score = -logit
        else:
            score = -logit + 1000
        
        return score, 0

    def update_colors_from_matches(self, colors, corners, edges, centers, corner_matches, edge_matches, center_matches):
        """
        Updates the colors array based on the matches found.
        """
        # Update corners
        for extracted_idx, legal_idx, rotation in corner_matches:
            extracted_piece = corners[extracted_idx]
            legal_piece = LEGAL_PIECES["corners"][legal_idx]
            
            # Update each sticker color
            for sticker_idx in range(3):
                face_idx, pos_idx = extracted_piece.indices[sticker_idx]
                legal_color = legal_piece[(sticker_idx + rotation) % 3]
                colors[face_idx][pos_idx] = legal_color
        
        # Update edges
        for extracted_idx, legal_idx, rotation in edge_matches:
            extracted_piece = edges[extracted_idx]
            legal_piece = LEGAL_PIECES["edges"][legal_idx]
            
            for sticker_idx in range(2):
                face_idx, pos_idx = extracted_piece.indices[sticker_idx]
                legal_color = legal_piece[(sticker_idx + rotation) % 2]
                colors[face_idx][pos_idx] = legal_color
        
        # Update centers
        for extracted_idx, legal_idx, _ in center_matches:
            extracted_piece = centers[extracted_idx]
            legal_piece = LEGAL_PIECES["centers"][legal_idx]
            
            face_idx, pos_idx = extracted_piece.indices[0]
            colors[face_idx][pos_idx] = legal_piece
        
        return colors

    def get_stickers_to_switch(self, color, colors, color_probs, col_corrections):
        SIMILAR = [4, 3, 5, 1, 0, 2]
        possibilities = []
        check_depth = 2
        finished_search = False
        while not finished_search and check_depth < 9:
            # Get possible stickers to switch, and what to switch to
            for f in range(len(colors)):
                for c in range(len(colors[f])):
                    col = colors[f][c]
                    if col == color:
                        next_best_col, next_best_prob = self.get_col_and_prob_from_pos(color_probs, f, c, 2)
                        if self.is_col_lacking(next_best_col, col_corrections):
                            if len(possibilities) < 1:
                                possibilities.append([f, c, next_best_col, next_best_prob])
                            else:
                                insert_index = 0
                                for i in range(len(possibilities)):
                                    insert_index = i
                                    if possibilities[i][3] < next_best_prob:
                                        print("PLACEHOLDER")
            check_depth += 1
        # Find switch to a Similar color, or the greatest probability otherwise
        # for p in range(len(possibilities)):
        #     if possibilities[p][2] == SIMILAR[color]:


    def is_col_lacking(self, col, col_corrections):
        return col_corrections[col] < 0

    def get_col_and_prob_from_pos(self, color_probs, face_index, sticker_index, rank=1):
        # rank = which probability in the "best ranking list" to return. 1 = very best, 2 = second, etc.
        probs = color_probs[face_index][sticker_index].copy()
        best_index = probs.argmax().item()
        best_prob = probs[best_index]
        for i in range(rank-1):
            probs[best_index] = -99
            best_index = probs.argmax().item()
            best_prob = probs[best_index]
        # Returns color [0-5] and corresponding prob (float)
        return best_index, best_prob


    def get_col_corrections(self, colors):
        # Counts and returns how many extra or too few of each color there are
        corrections = [0, 0, 0, 0, 0, 0]
        for face in colors:
            for c in face:
                if c < 6: corrections[c] += 1
        for i in range(len(corrections)):
            corrections[i] -= 9
        return corrections

    def get_cols_from_probs(self, color_probs):
        # color_probs is a 6x9x6 nested list
        colors = []
        for face_index in range(len(color_probs)):
            print(f"Reading face {face_index} colors")
            face = []
            for sticker_index in range(len(color_probs[face_index])):
                probs = color_probs[face_index][sticker_index]
                max_prob = probs[0]
                max_index = 0
                for p in range(len(probs)):
                    if probs[p] > max_prob:
                        max_prob = probs[p]
                        max_index = p
                face.append(max_index)
            colors.append(face)
        return colors

    def ScanManyFacesData(self, start_index=0, start_state=None):
        # This method is used to scan and twist the rubik's cube many times to create a dataset of face image --> colors array.
        # EXPECTED: physical cube starts in solved state, in default orientation.
        cube = None
        if start_state is None:
            cube = CubeState()
        else:
            cube = start_state

        for i in range(200):
            self.append_current_face_labels(cube)
            self.ReadCube(record_data=True, optional_index=(i+start_index))
            moves = self.get_n_rand_moves(4)
            print(f"Scrambling: {moves}")
            for m in moves:
                self.claw_machine.move(m)
                cube.move(m)
                print(f"New cube state:")
                print(cube.flat_data)


    def append_current_face_labels(self, cubestate):
        faces_order = ['L', 'F', 'R', 'B', 'U', 'D']
        with open("UncroppedLabels.txt", 'a') as file:
            for i in range(len(faces_order)):
                face_to_write = faces_order[i]
                cubestate_face_index = FACE_SEQUENCE.index(face_to_write)
                face = cubestate.shaped_data[cubestate_face_index]
                face_code = ""
                for c in face:
                    face_code = face_code + str(c)
                face_code += "\n"
                file.write(face_code)

    def get_n_rand_moves(self, n):
        scramble_moves = ["R", "L", "F", "B"]
        scramble_types = ["", "'", "2"]
        moves = []
        for i in range(n):
            rand_move = scramble_moves[i%len(scramble_moves)]#scramble_moves[random.randint(0, len(scramble_moves)-1)]
            rand_type = scramble_types[random.randint(0, len(scramble_types)-1)]
            move = rand_move + rand_type
            moves.append(move)
        return moves

    def read_face(self, cropped_img):
        face = []
        for y in range(3):
            for x in range(3):
                left = x * 50
                upper = y * 50
                right = left + 50
                lower = upper + 50
                sticker = cropped_img.crop((left, upper, right, lower))
                sticker_color = self.read_sticker(sticker)
                face.append(sticker_color)



    def read_sticker(self, sticker):
        # sticker is a 2D array of color values (encoded into one number hopefully)
        print("temp read_sticker()")
        return 0

# reader = CubeReader(None)
# colors = reader.get_cols_from_probs(TestProbabilities)
# reader.extract_pieces(colors, TestProbabilities)