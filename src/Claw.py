import time, math
from adafruit_motor import servo
from PCABoard import PCABoard

class Claw:
    # Angles corresponding to the extended state of each claw.
    # These angles are when the claws fully push on the cube.
    extended_angles = {
        "L": 35,
        "F": 37,
        "R": 20,
        "B": 10,
        "D": 13
    }

    # NOTE: horizontal and vertical positions must be such that a 90-degree
    # turn clockwise is always possible after that position.
    horizontal_positions = {
        "L": 1,
        "F": 2,
        "R": 1,
        "B": 2,
        "D": 2
    }

    # (See note for horizontal_positions)
    vertical_positions = {
        "L": 2,
        "F": 1,
        "R": 2,
        "B": 1,
        "D": 1
    }

    def __init__(self, extendorChannel, twisterChannel, face):
        pca = PCABoard().get()
        #print("Init Claw "+face)

        self.extendor = servo.Servo(pca.channels[extendorChannel], min_pulse=500, max_pulse=2400)
        if face == "D":
            self.extendor.angle = Claw.extended_angles["D"]
        else:
            self.retract()

        self.twister = servo.Servo(pca.channels[twisterChannel], min_pulse=500, max_pulse=2400)
        self.angle = 90 # The target angle of the Twister only
        self.twister.angle = self.angle

        self.face = face

        time.sleep(0.5)

    def retract(self):
        self.extendor.angle = 160

    def extend(self, push=True):
        offset = 0
        if push == False:
            if self.face == "D":
                offset = -15
            else:
                offset = -20
        angle = Claw.extended_angles[self.face]
        self.extendor.angle = angle - offset

    def twist(self, position, doOffset=True, slow=True): # ADJUST: Default should be slow=False; keep it =True for now (testing)
        # position = 1, 2, or 3:  fully anticlockwise, halfway, or fully clockwise.
        if not (position == 1 or position == 2 or position == 3):
            print(f"Error: Cannot twist claw to position {position}! Must be 1, 2, or 3.")
            return

        target = 90 * (position - 1)

        # if doOffset:
        #     offset = math.copysign(20, (target-self.angle))
        #     self.set_angle(max(0, min(180, target+offset)), slow)
            #time.sleep(0.4)
        offset = 0
        if doOffset: offset = 20
        self.set_angle(target, offset, slow)
        #time.sleep(0.4)

    def set_angle(self, angle, offset=0, slow=True): # ADJUST: Default should be slow=False; keep it =True for now (testing)
        # Sets the Twister servo angle, and records it in self.angle. Also has a slow turn option (slow=True).
        #print(f"{angle} degrees, slow={slow}")
        if self.face == "F":
            angle = max(0, min(180, angle + 2))
            offset = offset + 3

        offsetAngle = angle + math.copysign(offset, (angle - self.angle))
        offsetAngle = max(0.0, min(180.0, offsetAngle))

        if slow:
            self.stepSlowly(offsetAngle)
            self.stepSlowly(angle)
        else:
            self.twister.angle = offsetAngle
            self.angle = offsetAngle
            # Don't return to target angle because relying on instant angle setting for turn_cube clockwise_90()
            # task.wait(0.5)
            # self.twister.angle = angle
            # self.angle = angle
            # task.wait(0.5)

    def stepSlowly(self, end_angle):
        # HELPER METHOD for set_angle()
        DEGREES_PER_SEC = 120.0  # For the slow=True option
        STEPS_PER_SEC = 100.0

        start_angle = self.angle
        #end_angle = offsetAngle
        cur_angle = start_angle

        step_dir = math.copysign(1.0, end_angle - start_angle)
        step = (DEGREES_PER_SEC / STEPS_PER_SEC) * step_dir

        while abs(cur_angle - end_angle) > abs(step * 2.0):
            cur_angle = max(min(cur_angle + step, 180), 0)
            self.twister.angle = cur_angle
            self.angle = cur_angle
            time.sleep(1.0 / STEPS_PER_SEC)

    def horizontal(self, doOffset=False, slow=False):
        position = Claw.horizontal_positions[self.face]
        self.twist(position, doOffset, slow)

    def vertical(self, doOffset=False, slow=False):
        position = Claw.vertical_positions[self.face]
        self.twist(position, doOffset, slow)

    def clockwise_90(self, offset=20, slow=True):
        self.set_angle(min(180, self.angle + 90), offset, slow)

    def anti_clockwise_90(self, offset=20, slow=True):
        self.set_angle(min(180, self.angle - 90), offset, slow)