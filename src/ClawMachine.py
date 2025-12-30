from Claw import Claw
import time

class ClawMachine:
    
    adjacent_faces = {
        "L": "F",
        "F": "R",
        "R": "B",
        "B": "L",
        "D": "F"
    }

    opposite_faces = {
        "L": "R",
        "F": "B",
        "R": "L",
        "B": "F"
    }
    
    def __init__(self):
        # ADJUST
        self.claws = {
            "D" : Claw(6, 7, "D"),
            "F" : Claw(9, 13, "F"),
            "L" : Claw(8, 12, "L"),
            "R" : Claw(10, 14, "R"),
            "B" : Claw(11, 15, "B"),
        }

        # center_cube() includes default_position()
        self.center_cube()

    def move(self, move):
        # move =  F, R', U2, etc.
        # This is the main interface for the most important method; performs a move on the cube.
        face = move[0]
        move_type = move[1:]

        print(f"MOVING!   Face: {face}   Move_Type: \"{move_type}\"")
        self.turn_face(face, move_type)
        time.sleep(1)

    def hold_cube(self, push=True):
        # Simple hold, only using L and R claws
        self.claws["L"].horizontal()
        self.claws["R"].vertical()
        time.sleep(0.5)
        self.claws["L"].extend(push=push)
        self.claws["R"].extend(push=push)
        time.sleep(0.7)

    def release_cube(self):
        # Simply release the hold_cube (only L and R claws)
        self.claws["L"].retract()
        time.sleep(0.7)
        self.claws["R"].retract()
        time.sleep(0.7)

    def turn_cube(self, face_move):
        # face_move = F, R, L, etc.
        # Turns the entire cube in the same direction as the given face turn.
        # Can only turn 90 degrees clockwise. Only give plain face codes, not moves (no ' or 2)

        # Steps:
        # Hold Cube.
            # Extend the claws on the face_move axis, and hold cube in opposite directions (horizontal & vertical)
        # Turn Cube
            # Simply twist the extended holder claws in the right direction once

        self.center_cube()

        if face_move == "D" or face_move == "U":

            self.hold_cube(push=True)
            time.sleep(1)

            self.claws["D"].retract()
            time.sleep(0.5)
            self.claws["D"].twist(2, False)
            time.sleep(0.5)
            self.claws["D"].extend()
            time.sleep(1)

            self.release_cube()

            if face_move == "D":
                self.claws["D"].twist(3, False) # ADJUST to achieve clockwise rot
            else:
                self.claws["D"].twist(1, False)  # ADJUST to achieve ANTI-clockwise rot

        else:

            # First, grab cube by extending without push, retracting D, then extending with push
                # (face_move and opposite_face claws should be opposite (vertical and horizontal)).
            # Then, rotate cube 90 degrees clockwise.

            opposite_face = ClawMachine.opposite_faces[face_move]
            adjacent_face1 = ClawMachine.adjacent_faces[face_move]
            adjacent_face2 = ClawMachine.opposite_faces[adjacent_face1]

            self.claws[adjacent_face1].vertical()
            self.claws[adjacent_face2].vertical()

            # Hold cube tightly
            self.claws[face_move].twist(2, doOffset=False, slow=False)
            self.claws[opposite_face].twist(3, doOffset=False, slow=False)
            time.sleep(0.7)
            self.claws[adjacent_face1].extend(push=True)
            self.claws[adjacent_face2].extend(push=True)
            time.sleep(1)
            self.claws[face_move].extend(push=True)
            self.claws[opposite_face].extend(push=True)
            time.sleep(1)
            self.claws[adjacent_face1].retract()
            self.claws[adjacent_face2].retract()

            # Retract D
            self.claws["D"].retract()
            time.sleep(1)

            # if face_move == "L" or face_move == "R":
            #     self.claws["D"].twist(2, False)
            # else:
            #     self.claws["D"].twist(1, False)
            # self.claws["D"].extend()

            # Turn Cube
            self.claws[face_move].clockwise_90(offset=2, slow=False)
            self.claws[opposite_face].anti_clockwise_90(offset=2, slow=False)
            time.sleep(2)

            # Release
            vertical_claw = opposite_face if Claw.horizontal_positions[face_move] == 1 else face_move
            self.claws[vertical_claw].extend(push=False)

            time.sleep(0.5)
            self.claws["D"].extend(push=False)

            # Reset to default position
            time.sleep(0.8)
            self.claws[vertical_claw].retract()
            time.sleep(0.3)
            self.claws["D"].extend()
            time.sleep(0.7)
            self.claws[ClawMachine.opposite_faces[vertical_claw]].retract()

        time.sleep(1)

    def default_position(self):
        self.claws["D"].extend()
        self.claws["L"].retract()
        self.claws["R"].retract()
        time.sleep(0.5)
        self.claws["F"].retract()
        time.sleep(0.5)
        self.claws["B"].retract()
        time.sleep(0.4)

    def default_claws(self):
        #self.claws["D"].twist(2, doOffset=False, slow=False)
        self.claws["L"].twist(2, doOffset=False, slow=False)
        self.claws["F"].twist(2, doOffset=False, slow=False)
        self.claws["R"].twist(2, doOffset=False, slow=False)
        self.claws["B"].twist(2, doOffset=False, slow=False)
        time.sleep(0.4)

    def turn_face(self, face, move_type):
        # face = F, R, B, U, etc. (string)
        # move_type = "", "'", or "2" (string)

        # This is the main method to turn any face of the cube with any move_type.

        # Set adjacent face claw to vertical (horizontal if face = "D") and extend to hold
        # At same time, extend opposite claws (of face and adjacent) to prevent from falling
        # Finally, when done, extend and turn the claw in charge of "face"

        # If face == "U", simply turn_cube, turn_face(F), and turn_cube back.
        if face == "U":
            self.turn_cube("L")
            self.turn_face("F", move_type)
            self.turn_cube("R")
            return
        elif face == "D":
            self.turn_cube("R")
            self.turn_face("F", move_type)
            self.turn_cube("L")
            return

        # First, default position and set all claw angles
        self.default_position()
        opposite_face = ClawMachine.opposite_faces[face]
        adjacent_face1 = ClawMachine.adjacent_faces[face]
        adjacent_face2 = ClawMachine.opposite_faces[adjacent_face1]

        self.claws[adjacent_face1].vertical()
        self.claws[adjacent_face2].vertical()

        if move_type == "":
            self.claws[face].twist(1, doOffset=False, slow=False)
        elif move_type == "'":
            self.claws[face].twist(3, doOffset=False, slow=False)
        elif move_type == "2":
            self.claws[face].twist(1, doOffset=False, slow=False)
        else:
            print(f"WARNING: Unexpected move_type in turn_face()! ({move_type})")

        time.sleep(1)

        # If face != "D", the next step is to hold the cube tightly:
        self.claws[adjacent_face1].extend(push=True)
        self.claws[adjacent_face2].extend(push=True)
        time.sleep(0.7)
        self.claws[face].extend(push=True)
        self.claws[opposite_face].extend(push=True)
        time.sleep(0.7)
        self.claws[adjacent_face2].extend(push=False)


        # Then retract D:
        self.claws["D"].retract()
        time.sleep(1)

        # Then rotate the face.
        self.claws[face].extend(push=False)
        time.sleep(0.4)
        if move_type == "":
            self.claws[face].twist(2, slow=True)
        elif move_type == "'":
            self.claws[face].twist(2, slow=True)
        elif move_type == "2":
            self.claws[face].twist(2, slow=True)
            self.claws[face].retract()
            time.sleep(0.5)
            self.claws[face].twist(1, slow=False)
            time.sleep(0.4)
            self.claws[face].extend(push=False)
            time.sleep(1)
            self.claws[face].twist(2, slow=True)
        else:
            print(f"WARNING: Unexpected move_type for turn_face()! ({move_type})")
        time.sleep(1)

        # Hold cube gently
        self.claws[face].extend(push=False)
        self.claws[opposite_face].extend(push=False)
        self.claws[adjacent_face1].extend(push=False)
        self.claws[adjacent_face2].extend(push=False)
        time.sleep(0.5)

        # Finally, reset to default position
        self.claws["D"].extend(push=False)
        time.sleep(1)

        self.claws[opposite_face].retract()
        self.claws[face].retract()
        time.sleep(0.7)
        self.claws[adjacent_face1].retract()
        self.claws["D"].extend(push=True)
        time.sleep(0.7)
        self.claws[adjacent_face2].retract()

        time.sleep(1)

    def center_cube(self, d_pos=2):
        self.default_position()
        self.default_claws()
        time.sleep(0.3)

        self.claws["L"].extend(push=True)
        self.claws["R"].extend(push=True)
        time.sleep(0.6)
        self.claws["F"].extend(push=True)
        self.claws["B"].extend(push=True)
        time.sleep(0.6)

        self.claws["D"].retract()
        time.sleep(0.3)
        self.claws["D"].twist(d_pos, doOffset=False, slow=False)
        time.sleep(0.3)
        self.claws["D"].extend()
        time.sleep(0.7)

        self.default_position()
        
    def face_to_cam(self, face_num):
        # face_num = 0, 1, 2, 3, 4, or 5.  face_num = 6 --> finish & reset cube position
        # shows a face in the correct orientation.
        # ASSUMING SPECIFIC INITIAL CUBE ORIENTATION:
        # U = Yellow, F = Green, L = Red
        # ALSO, ASSUMING that this will be called with face_num = 0, 1, 2, etc. one after the other.
        if face_num == 0:
            self.center_cube(d_pos=1)
            self.claws["L"].vertical()
            self.claws["F"].vertical()
            self.claws["R"].vertical()
            self.claws["B"].vertical()

            self.claws["D"].set_angle(self.claws["D"].angle+45, offset=0, slow=True)
        elif face_num < 4:
            self.claws["D"].set_angle(self.claws["D"].angle + 45, offset=0, slow=True)
            self.hold_cube(push=True)
            self.claws["D"].retract()
            time.sleep(0.3)
            self.claws["D"].twist(1, doOffset=False, slow=False)
            time.sleep(0.3)
            self.claws["D"].extend()
            time.sleep(0.7)
            self.release_cube()
            self.claws["D"].set_angle(self.claws["D"].angle + 45, offset=0, slow=True)
        elif face_num == 4:
            self.claws["D"].set_angle(self.claws["D"].angle + 45, offset=0, slow=True)
            self.turn_cube("L")
            self.turn_cube("U")
            self.claws["D"].set_angle(self.claws["D"].angle + 45, offset=0, slow=True)
        elif face_num == 5:
            self.claws["D"].set_angle(self.claws["D"].angle + 45, offset=0, slow=True)
            self.turn_cube("R")
            self.turn_cube("R")
            self.claws["D"].set_angle(self.claws["D"].angle - 45, offset=0, slow=True)
        elif face_num == 6:
            # RESET CUBE POSITION
            self.claws["D"].set_angle(self.claws["D"].angle + 45, offset=0, slow=True)
            self.turn_cube("R")




    def next_D_45(self):
        print("temp")


        

    time.sleep(3)
