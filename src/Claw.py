import time, math
from adafruit_motor import servo
from PCABoard import PCABoard

class Claw:
    def __init__(self, extendorChannel, twisterChannel, face):
        pca = PCABoard().get()

        # Angles corresponding to the extended state of each claw
        self.extended_angles = {
            "L": 55,
            "F": 50,
            "R": 30,
            "B": 30,
            "D": 14
        }

        self.extendor = servo.Servo(pca.channels[extendorChannel], min_pulse=500, max_pulse=2400)
        if face == "D":
            self.extendor.angle = self.extended_angles["D"]
        else:
            self.extendor.angle = 160

        self.twister = servo.Servo(pca.channels[twisterChannel], min_pulse=500, max_pulse=2400)
        self.angle = 90 # The target angle of the Twister only
        self.twister.angle = self.angle

        self.face = face

        # NOTE: horizontal and vertical positions must be such that a 90-degree
        # turn clockwise is always possible after that position.
        self.horizontal_positions = {
            "L" : 1,
            "F": 2,
            "R": 1,
            "B": 2,
            "D": 2
        }

        # (See note for horizontal_positions)
        self.vertical_positions = {
            "L" : 2,
            "F": 1,
            "R": 2,
            "B": 1,
            "D": 1
        }

        time.sleep(0.5)

    def retract(self):
        self.extendor.angle = 160

    def extend(self, push=True):
        offset = 0
        if push and self.face != "D": offset = 15
        angle = self.extended_angles[self.face]
        self.extendor.angle = angle - offset

    def twist(self, position, doOffset=True, slow=True): # ADJUST: Default should be slow=False; keep it =True for now (testing)
        # position = 1, 2, or 3:  fully anticlockwise, halfway, or fully clockwise.
        if not (position == 1 or position == 2 or position == 3):
            print(f"Error: Cannot twist claw to position {position}! Must be 1, 2, or 3.")
            return

        target = 90 * (position - 1)

        if doOffset:
            offset = math.copysign(15, (target-self.angle))
            self.set_angle(max(0, min(180, target+offset)), slow)
            #time.sleep(0.4)
        else:
            self.set_angle(target, slow)
            #time.sleep(0.4)

    def set_angle(self, angle, slow=True): # ADJUST: Default should be slow=False; keep it =True for now (testing)
        # Sets the Twister servo angle, and records it in self.angle. Also has a slow turn option (slow=True).
        print(f"{angle} degrees, slow={slow}")
        if slow:
            DEGREES_PER_SEC = 90.0  # For the slow=True option
            STEPS_PER_SEC = 100.0

            start_angle = self.angle
            end_angle = angle
            cur_angle = start_angle

            step_dir = math.copysign(1.0, end_angle - start_angle)
            step = (DEGREES_PER_SEC / STEPS_PER_SEC) * step_dir

            while abs(cur_angle - end_angle) > abs(step*3.0):

                cur_angle = max(min(cur_angle + step, 180), 0)
                self.twister.angle = cur_angle
                self.angle = cur_angle
                time.sleep(1.0/STEPS_PER_SEC)

        self.twister.angle = angle
        self.angle = angle

    def horizontal(self, doOffset=False, slow=False):
        position = self.horizontal_positions[self.face]
        self.twist(position, doOffset, slow)

    def vertical(self, doOffset=False, slow=False):
        position = self.vertical_positions[self.face]
        self.twist(position, doOffset, slow)

    def clockwise_90(self, slow=True):
        self.set_angle(min(180, self.angle + 90), slow)

    def anti_clockwise_90(self, slow=True):
        self.set_angle(min(180, self.angle - 90), slow)