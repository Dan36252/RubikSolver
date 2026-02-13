# Importing Google Drive folder
# from google.colab import drive
# drive.mount('/content/drive')
# import sys
# sys.path.insert(0, '/content/drive/MyDrive/RubikSolver/src')

import numpy as np
import re
from CubeTurnMaps import MAPS, get_index_from_code

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

SOLVED_STATE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5
]

def swap_faces(arr, i1, i2):
    # arr: numpy array of shape (6, 9)
    temp = arr[i1].copy()
    arr[i1] = arr[i2].copy()
    arr[i2] = temp

def shape_data(flat_data):

    if len(flat_data) != 54:
        print(f"WARNING: data is len {len(flat_data)}, but 54 is expected")
        if len(flat_data) < 54: return

    shaped_data = []
    for f in range(6):
        face = []
        for c in range(9):
            i = f*9 + c
            face.append(flat_data[i])
        shaped_data.append(face)

    return shaped_data

def flatten_data(shaped_data):
    flat_data = []

    for f in range(len(shaped_data)):
        for c in range(len(shaped_data[f])):
            flat_data.append(shaped_data[f][c])

    return flat_data


class CubeState:
    # Mechanics:
    # - Input Data (54x1)
    # - Perform Moves
    # - Save previous moves
    # - Output Data (54x1)
    # - Output Previous Moves ((30*3)x1)

    def __init__(self, data=SOLVED_STATE):
        # data: a numpy array of shape (54,), representing a cube state (following rules above)
        self.flat_data = data
        self.shaped_data = shape_data(data)

    def get_flat_data(self):
        return self.flat_data

    def get_shaped_data(self):
        return self.shaped_data

    def standardize_data(self):
        for i in range(len(self.shaped_data)):
            this_face = self.shaped_data[i][4]
            if this_face != i:
                for j in range(len(self.shaped_data)-i-1):
                    if self.shaped_data[i+j+1][4] == i:
                        swap_faces(self.shaped_data, i, i+j+1)
        self.flat_data = flatten_data(self.shaped_data)

    def move(self, move_letter):
        # move_letter can be R, L', U2, etc.
        turn_map = MAPS[move_letter]
        mappings = turn_map.split()
        print(f"Move: {move_letter}, Mappings Length: {len(mappings)}, previous state:")
        print(self.flat_data)
        new_data = [0] * 54
        for m in range(len(mappings)):
            # Replace each color value in current data (0-53) with the color at the new index
            new_index = get_index_from_code(mappings[m])
            if new_index == -1: new_index = m
            new_data[m] = self.flat_data[new_index]
        print("New state:")
        print(new_data)
        self.flat_data = new_data
        self.shaped_data = shape_data(self.flat_data)

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



    def __repr__(self):
        str = "Cube:\n"
        str += "".join(self.shaped_data[0]) + "\n"
        str += "".join(self.shaped_data[1]) + "\n"
        str += "".join(self.shaped_data[2]) + "\n"
        str += "".join(self.shaped_data[3]) + "\n"
        str += "".join(self.shaped_data[4]) + "\n"
        str += "".join(self.shaped_data[5]) + "\n"
        return str

