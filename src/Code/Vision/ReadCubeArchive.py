from sys import hash_info

from Camera import Camera
from PIL import Image
from CubeState import COLOR_SEQUENCE, FACE_SEQUENCE, CubeState
import time, cv2, random
import numpy as np
from VisionRunner import Model

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

        print("Running clean-up algorithm on predictions...")
        colors = self.get_cols_from_probs(color_probs) # A 6x9 list
        col_corrections = self.get_col_corrections(colors) # A len 6 list. Its elements should add up to 0.
        colors = self.correct_colors(colors, color_probs, col_corrections)

        return colors

    def correct_colors(self, colors, color_probs, col_corrections):

        # Create logic to only correct colors that have positive col_corrections. Negatives are taken care of automatically.

    def correct_one_color(self, color, colors, color_probs, col_corrections):
        if col_corrections[color] <= 0:
            print("WARNING: correct_one_color cannot perform corrections on negatives! Only run it on positives. Negatives are taken care of automatically.")
            return
        # TODO: Finish this method, which looks at all stickers of the given color, and finds the best sticker to switch and the best color to switch it to, and performs the switch.
        # You can only switch it to a color that is lacking (as seen in col_corrections).
        # The sticker of 'color' to be switched is the one whose next-best color probability is one of the lacking colors.
        # If multiple such stickers exist, perform the switch on the one with the best priority ranking:
        #   1) The sticker's next-best color is its SIMILAR color (red <-> orange, yellow <-> white, etc.)
        #   2) The sticker's next-best color probability is the highest

        # Can use this loop to check which stickers can be switched for example.
        # But organize code into helper methods, including this loop.


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

            check_depth += 1
        # Find switch to a Similar color, or the greatest probability otherwise
        for p in range(len(possibilities)):
            if possibilities[p][2] == SIMILAR[color]:


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