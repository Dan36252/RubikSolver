import numpy as np

# Faces Order: R, Y, G, W, O, B. (Corresponding color indices: R=0, Y=1, ...)
# Default Orientation: Red left, Yellow up, Green front, White down, Orange right, Blue back
# Unwrapping: From Default Orientation into cube cross, long side right.
# Data Flatten: From Unwrapped "Cube-Cross," take each face in order (R, Y, G, ...) and
# read the colors from left to right, top down. Flatten into a 1-D array.

COLOR_SEQUENCE = ['r', 'y', 'g', 'w', 'o', 'b']
FACE_SEQUENCE = ['L', 'U', 'F', 'D', 'R', 'B']
MOVE_SEQUENCE = []
for f in FACE_SEQUENCE:
    MOVE_SEQUENCE.append(f)
    MOVE_SEQUENCE.append(f+"'")
    MOVE_SEQUENCE.append(f+"2")
MOVE_SEQUENCE.append('#')

def swap_faces(arr, i1, i2):
    # arr: numpy array of shape (6, 9)
    temp = arr[i1].copy()
    arr[i1] = arr[i2].copy()
    arr[i2] = temp

class CubeState():
    def __init__(self, data):
        # data: a numpy array of shape (6, 9), representing 6 faces with 9 colors each
        self.full_data = data
        self.standardize_data()

    def standardize_data(self):
        for i in range(len(self.full_data)):
            this_face = self.full_data[i][4]
            if this_face != i:
                for j in range(len(self.full_data)-i-1):
                    if self.full_data[i+j+1][4] == i:
                        swap_faces(self.full_data, i, i+j+1)

    def __repr__(self):
        str = "Cube:\n"
        str += np.array2string(self.full_data[0]) + "\n"
        str += np.array2string(self.full_data[1]) + "\n"
        str += np.array2string(self.full_data[2]) + "\n"
        str += np.array2string(self.full_data[3]) + "\n"
        str += np.array2string(self.full_data[4]) + "\n"
        str += np.array2string(self.full_data[5]) + "\n"
        return str

