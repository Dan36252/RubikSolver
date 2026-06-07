# Importing Google Drive folder
# from google.colab import drive
# drive.mount('/content/drive')
# import sys
# sys.path.insert(0, '/content/drive/MyDrive/RubikSolver/src')

import numpy as np
import re, copy, math
from Code.Data.CubeTurnMaps import MAPS, get_index_from_code
from Code.Data.CubeExtractor import CubeExtractor

# Faces Order: R, Y, G, W, O, B. (Corresponding color indices: R=0, Y=1, ...)
# Default Orientation: Red left, Yellow up, Green front, White down, Orange right, Blue back
# Unwrapping: From Default Orientation into cube cross, long side right.
# Data Flatten: From Unwrapped "Cube-Cross," take each face in order (R, Y, G, ...) and
# read the colors from left to right, top down. Flatten into a 1-D array.

COLOR_SEQUENCE = ['r', 'y', 'g', 'w', 'o', 'b']
FACE_SEQUENCE = ['L', 'U', 'F', 'D', 'R', 'B']

MOVE_SEQUENCE = ['-']
for f in FACE_SEQUENCE:
    MOVE_SEQUENCE.append(f)
    MOVE_SEQUENCE.append(f+"'")
    MOVE_SEQUENCE.append(f+"2")
MOVE_SEQUENCE.append('#')

LEGAL_PIECES = {
    "corners" : [[3, 0, 2], [3, 5, 0], [3, 4, 5], [3, 2, 4], [1, 0, 5], [1, 2, 0], [1, 4, 2], [1, 5, 4]],
    "edges" : [[3, 0], [3, 5], [3, 4], [3, 2], [0, 2], [0, 5], [4, 5], [4, 2], [1, 0], [1, 2], [1, 4], [1, 5]],
    "centers" : [0, 1, 2, 3, 4, 5]
}

# These are the sticker IDs that correspond to the colors in each of the LEGAL_PIECES above
DEEPCUBE_PIECES_TO_STICKERS = {
    "corners" : [[27, 8, 24], [33, 53, 6], [35, 44, 51], [29, 26, 42], [9, 0, 47], [15, 18, 2], [17, 36, 20], [11, 45, 38]],
    "edges" : [[30, 7], [34, 52], [32, 43], [28, 25], [5, 21], [3, 50], [41, 48], [39, 23], [12, 1], [16, 19], [14, 37], [10, 46]],
    "centers" : [4, 13, 22, 31, 40, 49]
}

SOLVED_STATE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5
]

class CubeState:
    # Mechanics:
    # - Input Data (54x1)
    # - Perform Moves
    # - Save previous moves
    # - Output Data (54x1)
    # - Output Previous Moves ((30*3)x1)

    def __init__(self, data=SOLVED_STATE, encode=True, prev_state=None, prev_move=None, g_cost=0):
        # data: a list of length 54, representing a cube state (following rules above)
        self.encode = encode
        self.prev_state = prev_state
        self.prev_move = prev_move
        self.g_cost = g_cost
        self.total_cost = 1e10

        is_data_deepcube = CubeState.is_data_deepcube(data)
        self.flat_data = data if not is_data_deepcube else CubeState.deepcube_to_flat(data)
        self.shaped_data = CubeState.shape_data(self.flat_data)
        if encode: self.encoded_data = CubeState.encode_data(self.shaped_data)
        #if encode: print(self.encoded_data)

    def __eq__(self, other):
        if type(other) != CubeState: return
        return self.flat_data == other.flat_data

    def get_flat_data(self):
        return self.flat_data

    def get_shaped_data(self):
        return self.shaped_data

    def get_encoded_data(self):
        if self.encoded_data is None:
            print("WARNING: Could not return Encoded data from CubeState! It does not exist. Returning None.")
            return None
        else:
            return self.encoded_data

    @staticmethod
    def get_deepcube_data(flat_data, shaped_data):
        #print(f"get_deepcube_data() flat input:")
        #print(flat_data)
        # Extract Pieces from Shaped Data
        extractor = CubeExtractor()
        corners_ext, edges_ext, centers_ext = extractor.extract_pieces(shaped_data)

        # Create lists to store corresponding converted sticker IDs
        final_stickers = [0]*54

        # Fill the final_stickers list based on the legal pieces that each extracted piece matches, and the extracted pieces' orientation
        CubeState._add_pieces_to_deepflat(corners_ext, "corners", final_stickers)
        CubeState._add_pieces_to_deepflat(edges_ext, "edges", final_stickers)
        CubeState._add_pieces_to_deepflat(centers_ext, "centers", final_stickers)

        return final_stickers

    @staticmethod
    def is_data_deepcube(flat_data):
        for c in flat_data:
            if c >= 6:
                return True
        return False

    @staticmethod
    def deepcube_to_flat(data):
        flat_data = []
        for c in data:
            flat_data.append(math.floor(c/9))
        return flat_data

    @staticmethod
    def _shaped_index_to_flat(shaped_index):
        return (9*shaped_index[0])+shaped_index[1]

    @staticmethod
    def _add_pieces_to_deepflat(extracted_pieces, pieces_type, final_stickers):
        for piece in extracted_pieces:
            legal_index, matching_l = CubeState.get_matching_legal_piece(piece, LEGAL_PIECES[pieces_type])
            p_orient = CubeState.get_piece_orientation(piece, matching_l)
            #print(p_orient)
            matching_dstickers = DEEPCUBE_PIECES_TO_STICKERS[pieces_type][legal_index]
            for s in range(piece.piece_type):
                adjusted_index = (s + p_orient) % piece.piece_type
                #if p_orient == 2: print(adjusted_index)
                flat_index = CubeState._shaped_index_to_flat(piece.indices[s])
                dsticker_id = matching_dstickers[adjusted_index] if type(matching_dstickers) == list else matching_dstickers
                final_stickers[flat_index] = dsticker_id

    def set_total_cost(self, total_cost):
        self.total_cost = total_cost

    @staticmethod
    def standardize_data(shaped_data):
        # Unused method; supposed to rearrange the faces of a shaped_data 2d list, and return the corresponding flat data
        for i in range(len(shaped_data)):
            this_face = shaped_data[i][4]
            if this_face != i:
                for j in range(len(shaped_data)-i-1):
                    if shaped_data[i+j+1][4] == i:
                        CubeState.swap_faces(shaped_data, i, i+j+1)
        flat_data = CubeState.flatten_data(shaped_data)
        return flat_data

    def move(self, move_letter):
        # move_letter can be R, L', U2, etc.
        turn_map = MAPS[move_letter]
        mappings = turn_map.split()
        # print(f"Move: {move_letter}, Mappings Length: {len(mappings)}, previous state:")
        # print(self.flat_data)
        new_data = [0] * 54
        for m in range(len(mappings)):
            # Replace each color value in current data (0-53) with the color at the new index
            new_index = get_index_from_code(mappings[m])
            if new_index == -1: new_index = m
            new_data[m] = self.flat_data[new_index]
        # print("New state:")
        # print(new_data)
        self.flat_data = new_data
        self.shaped_data = CubeState.shape_data(self.flat_data)
        if self.encode: self.encoded_data = CubeState.encode_data(self.shaped_data)

        # face_match = re.search("[A-Z]", move_letter)
        # if face_match != None:
        #     match_end_pos = face_match.end()
        #     face_to_move = move_letter[:match_end_pos] # L, R, U, etc
        #     move_type = move_letter[match_end_pos:]
        #     print(f"Face to move: {face_to_move}")
        #     print(f"Move type: {move_type}")
        #     face_index = FACE_SEQUENCE.index(face_to_move)
        #     if move_type == '':
        #         print("Moving normally!!!")
        #         self.full_data[face_index] = rotate_face(self.full_data[face_index])
        #     elif move_type == '\'':
        #         for i in range(3):
        #             self.full_data[face_index] = rotate_face(self.full_data[face_index])
        #     elif move_type == '2':
        #         for i in range(2):
        #             self.full_data[face_index] = rotate_face(self.full_data[face_index])

    def spawn_move(self, move_letter):
        # This returns a new CubeState with the applied move.
        # move_letter can be R, L', U2, etc.
        new_cubestate = CubeState(data=copy.deepcopy(self.flat_data), encode=True, prev_state=self, prev_move=move_letter, g_cost=self.g_cost+1)
        new_cubestate.move(move_letter)
        return new_cubestate

    @staticmethod
    def swap_faces(arr, i1, i2):
        # arr: numpy array of shape (6, 9)
        temp = arr[i1].copy()
        arr[i1] = arr[i2].copy()
        arr[i2] = temp

    @staticmethod
    def shape_data(flat_data):

        if len(flat_data) != 54:
            print(f"WARNING: data is len {len(flat_data)}, but 54 is expected")
            if len(flat_data) < 54: return

        shaped_data = []
        for f in range(6):
            face = []
            for c in range(9):
                i = f * 9 + c
                face.append(flat_data[i])
            shaped_data.append(face)

        return shaped_data

    @staticmethod
    def flatten_data(shaped_data):
        flat_data = []

        for f in range(len(shaped_data)):
            for c in range(len(shaped_data[f])):
                flat_data.append(shaped_data[f][c])

        return flat_data

    @staticmethod
    def encode_data(shaped_data):
        # Extract Pieces from Shaped Data
        extractor = CubeExtractor()
        corners_ext, edges_ext, centers_ext = extractor.extract_pieces(shaped_data)

        # Match each extracted piece to the corresponding standard piece (in LEGAL_PIECES) and calculate piece vector + orientation
        enc_corners = CubeState.encode_pieces(corners_ext, LEGAL_PIECES["corners"])
        enc_edges = CubeState.encode_pieces(edges_ext, LEGAL_PIECES["edges"])
        enc_centers = CubeState.encode_pieces(centers_ext, LEGAL_PIECES["centers"])

        # Combine encoded pieces into one vector
        enc_list = []
        for codes in [enc_corners, enc_edges, enc_centers]:
            for code in codes:
                for n in range(len(code)):
                    enc_list.append(float(code[n]))
                    #enc_string = enc_string + str(code[n]) + ", "
        #enc_string = enc_string[:-2] # Remove last comma
        #print(len(enc_string.split(",")))
        return enc_list # The encoded data is Length 104.



    face_vectors = [[-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [1, 0, 0], [0, 0, -1]]
    sticker_vectors = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]
    face_x_dirs = [[0,0,1], [1,0,0], [1,0,0], [1,0,0], [0,0,-1], [-1,0,0]]
    face_y_dirs = [[0,1,0], [0,0,-1], [0,1,0], [0,0,1], [0,1,0], [0,1,0]]

    @staticmethod
    def get_matching_legal_piece(vision_piece, legal_pieces):
        for i, l_piece in enumerate(legal_pieces):
            l_set = set(l_piece) if type(l_piece) == list else l_piece
            v_set = set(vision_piece.colors) if len(vision_piece.colors) > 1 else vision_piece.colors[0]
            if l_set == v_set:
                return i, l_piece
        raise Exception(f"Could not match a legal piece to this vision piece: {vision_piece.colors}")

    @staticmethod
    def encode_pieces(extracted_pieces, legal_pieces):
        #print("Encoding pieces.")
        codes = []
        for legal_c in legal_pieces:
            legal_set = set(legal_c) if type(legal_c) == list else legal_c
            #print(legal_set)
            for piece in extracted_pieces:
                extracted_set = set(piece.colors) if len(piece.colors) > 1 else piece.colors[0]
                if legal_set == extracted_set:
                    # This extracted piece corresponds to this legal piece.
                    piece_pos = CubeState.piece_to_vector(piece)
                    piece_orient = CubeState.get_piece_orientation(piece, legal_c) - 1 # Subtract 1 to standardize
                    code = np.concat((piece_pos, np.array([piece_orient])))
                    codes.append(code)
                    break
        return codes

    @staticmethod
    def piece_to_vector(piece):
        try:
            sticker_vectors = []
            for index in piece.indices:
                face_vector = np.array(CubeState.face_vectors[index[0]]) * 1.5
                sticker_x_comp = np.array(CubeState.face_x_dirs[index[0]]) * CubeState.sticker_vectors[index[1]][0]
                sticker_y_comp = np.array(CubeState.face_y_dirs[index[0]]) * CubeState.sticker_vectors[index[1]][1]
                sticker_vector = sticker_x_comp + sticker_y_comp + face_vector
                sticker_vectors.append(sticker_vector)
            piece_vector = np.zeros(len(sticker_vectors[0]))
            for i in range(len(sticker_vectors)):
                piece_vector = piece_vector + sticker_vectors[i]
            piece_vector = piece_vector / len(sticker_vectors)
            return piece_vector

        except Exception as e:
            raise Exception(f"Could not convert piece to vector while encoding data! Do indices exist? Indices: {piece.indices}  Error: {str(e)}")

    @staticmethod
    def get_piece_orientation(piece, reference):
        if piece.piece_type is None or piece.piece_type <= 0: raise Exception(f"Can't orient: Piece has a piece_type of {piece.piece_type}!")
        orient = 0
        while orient < piece.piece_type:
            matches = True
            for i in range(piece.piece_type):
                reference_color = reference[(i + orient) % piece.piece_type] if type(reference) == list else reference
                piece_color = piece.colors[i]
                #reference_color = reference[i] if type(reference) == list else reference
                if piece_color != reference_color:
                    matches = False
                    break
            if matches: return orient

            orient = orient + 1

        raise Exception(f"Could not orient piece: no matching orientations! Piece: {piece.colors}, Reference: {reference}")



    def __repr__(self):
        # str = "Cube:\n"
        # str += "".join(self.shaped_data[0]) + "\n"
        # str += "".join(self.shaped_data[1]) + "\n"
        # str += "".join(self.shaped_data[2]) + "\n"
        # str += "".join(self.shaped_data[3]) + "\n"
        # str += "".join(self.shaped_data[4]) + "\n"
        # str += "".join(self.shaped_data[5]) + "\n"
        return str(self.flat_data)


if __name__ == "__main__":
    state = CubeState()
    print(CubeState.get_deepcube_data(state.flat_data, state.shaped_data))
