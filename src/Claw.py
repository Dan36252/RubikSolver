import time
import math

class Claw:
    def __init__(self, extendorChannel, twisterChannel, face):
        # ADJUST
        self.extendor = servo.Servo(pca_channel[extendorChannel], min_pulse=-1, max_pulse=-1)
        self.twister = servo.Servo(pca_channel[twisterChannel], min_pulse=-1, max_pulse=-1)

        self.angle = 90 # The target angle of the Twister only
        self.face = face

        # ADJUST
        # NOTE: horizontal and vertical positions must be such that a 90-degree
        # turn clockwise is always possible after that position.
        self.horizontal_positions = {
            "L" : 1,
            "F": 2,
            "R": 1,
            "B": 2,
            "D": 2
        }

        # ADJUST
        # (See note for horizontal_positions)
        # For 'D', position 1 should be vertical.
        self.vertical_positions = {
            "L" : 2,
            "F": 1,
            "R": 2,
            "B": 1,
            "D": 1
        }

    def retract(self):
        self.extendor.angle = 160

    def extend(self):
        self.extendor.angle = 90

    def twist(self, position, doOffset=True):
        # position = 1, 2, or 3:  fully anticlockwise, halfway, or fully clockwise.
        if not (position == 1 or position == 2 or position == 3):
            print(f"Error: Cannot twist claw to position {position}! Must be 1, 2, or 3.")
            return
        target = 90 * (position - 1)
        if doOffset:
            offset = math.copysign(1, (90-self.angle)) * 15
            self.twister.angle = max(0, min(180, target+offset))
            time.sleep(0.5)
        self.twister.angle = target
        self.angle = target

    def horizontal(self, doOffset=False):
        position = self.horizontal_positions[self.face]
        self.twist(position, doOffset)

    def vertical(self, doOffset=False):
        position = self.vertical_positions[self.face]
        self.twist(position, doOffset)

    def clockwise_90(self):
        self.angle = min(180, self.angle + 90)
        self.twister.angle = self.angle