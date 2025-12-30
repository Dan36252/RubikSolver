from Camera import get_cropped_img
from PIL import Image
from CubeState import COLOR_SEQUENCE, CubeState

class CubeReader:

    def __init__(self, claw_machine):
        self.claw_machine = claw_machine

    # Procedure:
    # Simply have camera read every face (order doesn't matter) using computer vision.
    # Then, take the list of unordered faces data and order them according to Face Order (L, U, F, ...)
    # Finally, feed this into CubeState, which will be able to flatten it and perform "data cube turns."
    def ReadCube(self):
        # Returns: CubeState
        print("~~~")
        faces = []
        for i in range(6):
            self.claw_machine.face_to_cam(i)
            cropped_img = get_cropped_img()
            face = self.read_face(cropped_img)

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