from Code.Hardware.Claw import Claw
import time, atexit

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
        # Speed settings
        self.robot_speed = 2.0  # Default is 1.0; higher values make robot faster; smaller values make it slower
        self.fast_twist = True

        # Debug settings
        self.verbose = True

        # ADJUST
        self.claws = {
            "D" : Claw(6, 7, "D"),
            "F" : Claw(9, 13, "F"),
            "L" : Claw(8, 12, "L"),
            "R" : Claw(10, 14, "R"),
            "B" : Claw(11, 15, "B"),
        }

        # center_cube() includes default_position()
        self.centered = False
        self.center_cube()

        atexit.register(self.reset_D_claw)

    def move(self, move):
        # move =  F, R', U2, etc.
        # This is the main interface for the most important method; performs a move on the cube.
        face = move[0]
        move_type = move[1:]

        print(f"MOVING!   Face: {face}   Move_Type: \"{move_type}\"")
        self.turn_face(face, move_type)
        time.sleep(1)

    def hold_cube(self, push=True):
        if self.verbose: print("HOLD_CUBE()")
        # Simple hold, only using L and R claws
        self.claws["L"].horizontal()
        self.claws["R"].vertical()
        time.sleep(0.5/self.robot_speed)
        self.claws["L"].extend(push=push)
        self.claws["R"].extend(push=push)
        time.sleep(0.7/self.robot_speed)

    def release_cube(self):
        if self.verbose: print("RELEASE_CUBE()")
        # Simply release the hold_cube (only L and R claws)
        self.claws["L"].retract()
        time.sleep(0.7/self.robot_speed)
        self.claws["R"].retract()
        time.sleep(0.7/self.robot_speed)

    def turn_cube(self, face_move):
        if self.verbose: print("TURN_CUBE()")
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

            # self.hold_cube(push=True)
            # time.sleep(1)
            #
            # self.claws["D"].retract()
            # time.sleep(0.5)
            # self.claws["D"].twist(2, False)
            # time.sleep(0.5)
            # self.claws["D"].extend()
            # time.sleep(1)
            #
            # self.release_cube()

            if face_move == "D":
                self.claws["D"].twist(3, doOffset=False, slow=True) # ADJUST to achieve clockwise rot
            else:
                self.claws["D"].twist(1, doOffset=False, slow=True)  # ADJUST to achieve ANTI-clockwise rot

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
            time.sleep(0.7/self.robot_speed)
            self.claws[adjacent_face1].extend(push=True)
            self.claws[adjacent_face2].extend(push=True)
            time.sleep(1/self.robot_speed)
            self.claws[face_move].extend(push=True)
            self.claws[opposite_face].extend(push=True)
            time.sleep(1/self.robot_speed)
            self.claws[adjacent_face1].retract()
            self.claws[adjacent_face2].retract()

            # Retract D
            self.claws["D"].retract()
            time.sleep(0.7/self.robot_speed)

            # if face_move == "L" or face_move == "R":
            #     self.claws["D"].twist(2, False)
            # else:
            #     self.claws["D"].twist(1, False)
            # self.claws["D"].extend()

            # Turn Cube
            self.claws[face_move].clockwise_90(offset=2, slow=False)
            self.claws[opposite_face].anti_clockwise_90(offset=2, slow=False)
            time.sleep(0.5)

            # Release
            vertical_claw = opposite_face if Claw.horizontal_positions[face_move] == 1 else face_move
            self.claws[vertical_claw].extend(push=False)
            self.claws["D"].extend(push=False)
            time.sleep(0.3)
            self.claws[vertical_claw].retract()
            time.sleep(0.2)
            self.claws["D"].extend(push=True)
            time.sleep(0.3)
            self.claws[ClawMachine.opposite_faces[vertical_claw]].retract()

            self.centered = False

        # time.sleep(0.2)

    def default_position(self):
        if self.verbose: print("DEFAULT_POSITION()")
        self.claws["D"].extend()
        self.claws["L"].retract()
        self.claws["R"].retract()
        self.claws["F"].retract()
        self.claws["B"].retract()

    def reset_D_claw(self):
        self.claws["D"].twist(2, doOffset=False, slow=False)

    def default_claws(self):
        if self.verbose: print("DEFAULT_CLAWS()")
        #self.claws["D"].twist(2, doOffset=False, slow=False)
        self.claws["L"].twist(2, doOffset=False, slow=False)
        self.claws["F"].twist(2, doOffset=False, slow=False)
        self.claws["R"].twist(2, doOffset=False, slow=False)
        self.claws["B"].twist(2, doOffset=False, slow=False)

    def vertical_claws(self):
        if self.verbose: print("VERTICAL_CLAWS()")
        self.claws["L"].vertical()
        self.claws["F"].vertical()
        self.claws["R"].vertical()
        self.claws["B"].vertical()

    def turn_face(self, face, move_type):
        if self.verbose: print("TURN_FACE()")
        # face = F, R, B, U, etc. (string)
        # move_type = "", "'", or "2" (string)

        # This is the main method to turn any face of the cube with any move_type.

        # Set adjacent face claw to vertical (horizontal if face = "D") and extend to hold
        # At same time, extend opposite claws (of face and adjacent) to prevent from falling
        # Finally, when done, extend and turn the claw in charge of "face"

        # If face == "U" or "D", simply turn_cube, turn_face(F), and turn_cube back.
        if face == "U":
            if self.verbose: print("FACE == U")
            self.turn_cube("L")
            self.turn_face("F", move_type)
            self.turn_cube("R")
            return
        elif face == "D":
            if self.verbose: print("FACE == D")
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

        time.sleep(0.2)

        # Hold cube and prepare claws
        self.claws[adjacent_face1].extend(push=True)
        self.claws[adjacent_face2].extend(push=True)
        time.sleep(0.7/self.robot_speed)
        self.claws[face].extend(push=True)
        self.claws[opposite_face].extend(push=True)
        time.sleep(0.7/self.robot_speed)
        #self.claws[adjacent_face2].extend(push=False) # ?
        self.claws[face].extend(push=False)

        # Then retract D:
        self.claws["D"].retract()
        time.sleep(0.2)

        # Then rotate the face.
        if move_type == "":
            self.claws[face].twist(2, slow=(not self.fast_twist))
        elif move_type == "'":
            self.claws[face].twist(2, slow=(not self.fast_twist))
        elif move_type == "2":
            self.claws[face].twist(2, slow=(not self.fast_twist))
            self.claws[face].retract()
            time.sleep(0.2)
            self.claws[face].twist(1, slow=False)
            time.sleep(0.3)
            self.claws[face].extend(push=False)
            time.sleep(0.3)
            self.claws[face].twist(2, slow=(not self.fast_twist))
        else:
            print(f"WARNING: Unexpected move_type for turn_face()! ({move_type})")

        time.sleep(0.5/self.robot_speed)

        # Hold cube gently
        self.claws[face].extend(push=False)
        self.claws[opposite_face].extend(push=False)
        self.claws[adjacent_face1].extend(push=False)
        self.claws[adjacent_face2].extend(push=False)

        # Finally, reset to default position
        self.claws["D"].extend(push=False)
        time.sleep(0.3)

        self.claws[opposite_face].retract()
        self.claws[face].retract()
        self.claws[adjacent_face1].retract()
        self.claws["D"].extend(push=True)
        time.sleep(0.3)
        self.claws[adjacent_face2].retract()

        self.centered = False

    def center_cube(self, d_pos=2):
        if self.centered: return
        self.default_position()
        self.default_claws()
        time.sleep(0.2)

        self.claws["L"].extend(push=True)
        self.claws["R"].extend(push=True)
        time.sleep(0.2)
        self.claws["F"].extend(push=True)
        self.claws["B"].extend(push=True)
        time.sleep(0.2)

        self.claws["D"].retract()
        time.sleep(0.2)
        self.claws["D"].twist(d_pos, doOffset=False, slow=False)
        time.sleep(0.2)
        self.claws["D"].extend()
        time.sleep(0.2)

        self.default_position()
        time.sleep(0.3)
        self.centered = True
        
    def face_to_cam(self, face_num):
        # face_num = 0, 1, 2, 3, 4, or 5.  face_num = 6 --> finish & reset cube position
        # shows a face in the correct orientation.
        # ASSUMING SPECIFIC INITIAL CUBE ORIENTATION:
        # U = Yellow, F = Green, L = Red
        # ALSO, ASSUMING that this will be called with face_num = 0, 1, 2, etc. one after the other.
        if face_num == 0:
            self.center_cube(d_pos=3)
            self.claws["L"].vertical()
            self.claws["F"].vertical()
            self.claws["R"].vertical()
            self.claws["B"].vertical()

            self.claws["D"].set_angle(self.claws["D"].angle-45, offset=0, slow=True)
            self.centered = False
        elif face_num < 4:
            self.claws["D"].set_angle(self.claws["D"].angle - 45, offset=0, slow=True)
            self.center_cube(d_pos=3)
            self.claws["D"].set_angle(self.claws["D"].angle - 45, offset=0, slow=True)
            self.centered = False
        elif face_num == 4:
            self.claws["D"].set_angle(self.claws["D"].angle - 45, offset=0, slow=True)
            self.turn_cube("L")
            self.turn_cube("U")
            self.center_cube()
            self.claws["D"].set_angle(self.claws["D"].angle - 45, offset=0, slow=True)
            self.centered = False
        elif face_num == 5:
            self.claws["D"].set_angle(self.claws["D"].angle + 45, offset=0, slow=True)
            self.turn_cube("F")
            self.turn_cube("F")
            self.center_cube()
            self.claws["D"].set_angle(self.claws["D"].angle - 45, offset=0, slow=True)
            self.centered = False
        elif face_num == 6:
            # RESET CUBE POSITION
            self.claws["D"].set_angle(self.claws["D"].angle - 45, offset=0, slow=True)
            self.turn_cube("R")
            self.turn_cube("D")
            self.turn_cube("D")
            self.center_cube()
            return

        self.vertical_claws()




    def next_D_45(self):
        print("temp")


        

    time.sleep(3)
