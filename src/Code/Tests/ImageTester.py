#from Camera import Camera
#from VisionRunner import Model
from ReadCube import CubeReader
from RubikTestProbabilities import TestProbabilities
#from ClawMachine import ClawMachine
import time, cv2
import numpy as np

# runner = Model()
reader = CubeReader(None)
colors = reader.TestReadCube(TestProbabilities)
print(colors)

counts = [0, 0, 0, 0, 0, 0]

for f in range(len(colors)):
    for c in range(len(colors[f])):
        col = colors[f][c]
        if 0 <= col < 6:
            counts[col] += 1

print(counts)

# print("Ready")
# input("Press enter")
# img = cv2.imread("ALLFaceData/uncropped_img_0024-4.jpg")
# img = cv2.resize(img, (24, 24)) # Already done ine VisionRunner.py
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
# print("Read image")
# input("Press enter")
# colors_prediction = runner.predict(img)
# #print(colors_prediction)
# input("Press enter")


# Result with Standardization: [[0, 1, 5, 0, 3, 4, 4, 1, 1], [2, 3, 1, 2, 5, 0, 0, 3, 1], [3, 4, 5, 3, 0, 1, 4, 4, 3], [2, 2, 2, 5, 4, 5, 3, 3, 5], [0, 2, 4, 4, 2, 1, 0, 0, 3], [5, 2, 1, 0, 1, 3, 4, 5, 2]]
# Result without Standardization: [[2, 2, 0, 3, 3, 3, 3, 5, 4], [1, 3, 4, 1, 5, 0, 3, 5, 5], [5, 4, 0, 4, 0, 4, 2, 4, 2], [1, 2, 0, 1, 4, 0, 2, 3, 4], [1, 2, 5, 1, 2, 0, 3, 5, 5], [1, 2, 0, 1, 1, 0, 3, 5, 4]]