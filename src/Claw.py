import time, math
from adafruit_motor import servo
from PCABoard import PCABoard

class Claw:
    def __init__(self, extendorChannel, twisterChannel, face):
        pca = PCABoard().get()

        # ADJUST
        self.extendor = servo.Servo(pca.channels[extendorChannel], min_pulse=500, max_pulse=2400)
        self.extendor.angle = 160

        self.twister = servo.Servo(pca.channels[twisterChannel], min_pulse=500, max_pulse=2400)
        self.angle = 90 # The target angle of the Twister only
        self.twister.angle = self.angle

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

        # ADJUST
        # Angles corresponding to the extended state of each claw
        self.extended_angles = {
            "L" : 55,
            "F": 50,
            "R": 30,
            "B": 30,
            "D": 19
        }

        time.sleep(0.5)

    def retract(self):
        self.extendor.angle = 160

    def extend(self, push=True):
        offset = 0
        if push and self.face != "D": offset = 15
        angle = self.extended_angles[self.face]
        self.extendor.angle = angle - offset

    def twist(self, position, doOffset=True, slow=False):
        # position = 1, 2, or 3:  fully anticlockwise, halfway, or fully clockwise.
        if not (position == 1 or position == 2 or position == 3):
            print(f"Error: Cannot twist claw to position {position}! Must be 1, 2, or 3.")
            return

        target = 90 * (position - 1)

        if doOffset:
            offset = math.copysign(1, (90-self.angle)) * 15
            self.set_angle(max(0, min(180, target+offset)))
            time.sleep(0.8)

        self.set_angle(target)

    def set_angle(self, angle, slow=True): # ADJUST: Default should be slow=False; keep it =True for now (testing)
        # Sets the Twister servo angle, and records it in self.angle. Also has a slow turn option (slow=True).

        if slow:
            DEGREES_PER_SEC = 45  # For the slow=True option
            step = DEGREES_PER_SEC / 100

            start_angle = self.angle
            end_angle = angle
            cur_angle = start_angle
            while cur_angle < end_angle:
                cur_angle += step
                self.twister.angle = cur_angle
                time.sleep(0.01)

        self.twister.angle = angle
        self.angle = angle

    def horizontal(self, doOffset=False):
        position = self.horizontal_positions[self.face]
        self.twist(position, doOffset)

    def vertical(self, doOffset=False):
        position = self.vertical_positions[self.face]
        self.twist(position, doOffset)

    def clockwise_90(self):
        self.set_angle(min(180, self.angle + 90))