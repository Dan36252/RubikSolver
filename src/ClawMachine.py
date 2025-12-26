from Claw import Claw
import time

class ClawMachine:
    def __init__(self):
        # ADJUST
        self.claws = {
            "D" : Claw(0, 1, "D"),
            "F" : Claw(0, 1, "F"),
            "L" : Claw(0, 1, "L"),
            "R" : Claw(0, 1, "R"),
            "B" : Claw(0, 1, "B"),
        }

        self.claws["D"].extend()
        self.claws["R"].extend()
        self.claws["L"].extend()

        time.sleep(1.3)

        self.claws["R"].extend()
        self.claws["L"].extend()

        self.adjacent_faces = {
            "L" : "F",
            "F" : "R",
            "R" : "B",
            "B" : "L",
            "D" : "F"
        }

        self.opposite_faces = {
            "L": "R",
            "F": "B",
            "R": "L",
            "B": "F"
        }

    def hold_cube(self):
        # Simple hold, only using L and R claws
        self.claws["L"].horizontal()
        self.claws["R"].vertical()
        self.claws["L"].extend()
        self.claws["R"].extend()

    def release_cube(self):
        # Simply release the hold_cube (only L and R claws)
        self.claws["L"].retract()
        self.claws["R"].retract()

    def turn_cube(self, face_move):
        # face_move = F, R, L, etc.
        # Turns the entire cube in the same direction as the given face turn.
        # Can only turn 90 degrees clockwise. Only give plain face codes, not moves (no ' or 2)

        # Steps:
        # Hold Cube.
            # Extend the claws on the face_move axis, and hold cube in opposite directions (horizontal & vertical)
        # Turn Cube
            # Simply twist the extended holder claws in the right direction once

        if face_move == "D" or face_move == "U":

            self.hold_cube()

            self.claws["D"].retract()
            self.claws["D"].twist(2, False)
            self.claws["D"].extend()

            self.release_cube()

            if face_move == "D":
                self.claws["D"].twist(3, False) # ADJUST to achieve clockwise rot
            else:
                self.claws["D"].twist(1, False)  # ADJUST to achieve ANTI-clockwise rot

        else:

            # First, twist the D claw to not interfere with any of the axis claws
            opposite_face = self.opposite_faces[face_move]

            self.claws[face_move].horizontal()
            self.claws[opposite_face].horizontal()
            self.claws[face_move].extend()
            self.claws[opposite_face].extend()

            self.claws["D"].retract()
            if face_move == "L" or face_move == "R":
                self.claws["D"].twist(2, False)
            else:
                self.claws["D"].twist(1, False)
            self.claws["D"].extend()

            # Then, set axis claws to correct orientations
            self.claws[face_move].retract()
            self.claws[opposite_face].retraft()

            self.claws[face_move].horizontal()
            self.claws[opposite_face].vertical()

            self.claws[face_move].extend()
            self.claws[opposite_face].extend()

            # Finally, turn cube
            self.claws[face_move].clockwise_90()
            self.claws[opposite_face].clockwise_90()

            # Reset to default position
            self.claws["D"].extend()
            time.sleep(0.8)
            self.claws[face_move].retract()
            self.claws[opposite_face].retract()

    def single_turn_c(self, face):
        print("Single turn clockwise (placeholder)")
        # Set adjacent face claw to vertical (horizontal if face = "D") and extend to hold
        # At same time, extend opposite claws (of face and adjacent) to prevent from falling
        # Finally, when done, extend and turn the claw in charge of "face"
        adjacent_face = 