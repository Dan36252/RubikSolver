from Camera import Camera
from VisionRunner import Model
from ClawMachine import ClawMachine
import time, cv2
import numpy as np

cam = Camera()
runner = Model()
robot = ClawMachine()

robot.claws["B"].vertical()
robot.claws["F"].vertical()
robot.claws["D"].set_angle(135)

# LEFT OFF: trying to understand why CNN gives such bad predictions on the robot.
# Is colorspace consistent in cam reading and data reading (yes...?), is cropping good, and is lighting good? (FIX LIGHTING AND CROPPING)

c = 0
while c < 300:
    c += 1
    print("Ready")
    input("Press enter")
    img = cam.get_cropped_img().astype(np.uint8)
    #img = cv2.resize(img, (24, 24)) # Already done ine VisionRunner.py
    #test = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
    cv2.imshow("capture", img)
    cv2.waitKey(0)
    input("Press enter")
    colors_prediction = runner.predict_strict(img)
    #print(colors_prediction)
    input("Press enter")


