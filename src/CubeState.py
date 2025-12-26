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

# def swap_faces(arr, i1, i2):
#     # arr: numpy array of shape (6, 9)
#     temp = arr[i1].copy()
#     arr[i1] = arr[i2].copy()
#     arr[i2] = temp


class CubeState:
    # Mechanics:
    # - Input Data (54x1)
    # - Perform Moves
    # - Save previous moves
    # - Output Data (54x1)
    # - Output Previous Moves ((30*3)x1)

    def __init__(self, data):
        # data: a numpy array of shape (54,), representing a cube state (following rules above)
        self.data = data

    # def standardize_data(self):
    #     for i in range(len(self.full_data)):
    #         this_face = self.full_data[i][4]
    #         if this_face != i:
    #             for j in range(len(self.full_data)-i-1):
    #                 if self.full_data[i+j+1][4] == i:
    #                     swap_faces(self.full_data, i, i+j+1)

    def move(self, move_letter):
        # move_letter can be R, L', U2, etc.
        turn_map = MAPS[move_letter]
        mappings = turn_map.split()
        print(f"Mappings Length: {len(mappings)}")
        new_data = []
        for m in range(mappings):
            # Replace each color value in current data (0-53) with the color at the new index
            new_index = get_index_from_code(mappings[m])
            new_data[m] = self.data[new_index]
        print("New Data:")
        print(new_data)
        self.data = new_data

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
        str += np.array2string(self.full_data[0]) + "\n"
        str += np.array2string(self.full_data[1]) + "\n"
        str += np.array2string(self.full_data[2]) + "\n"
        str += np.array2string(self.full_data[3]) + "\n"
        str += np.array2string(self.full_data[4]) + "\n"
        str += np.array2string(self.full_data[5]) + "\n"
        return str

