# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2, time
from PIL import Image
import numpy as np

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

class Camera:

    def gstreamer_pipeline(
        self,
        sensor_id=0,
        capture_width=1920,
        capture_height=1080,
        display_width=960,
        display_height=540,
        framerate=30,
        flip_method=2,
    ):
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

    def read_camera(self):
        cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)

        # Let auto-exposure settle
        for _ in range(60):
            cap.read()

        ret, frame = cap.read()
        return frame

    def read_camera_old(self):
        window_title = "CSI Camera"

        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        print(gstreamer_pipeline(flip_method=0, sensor_id=1))
        video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if video_capture.isOpened():
            try:
                #window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
                while True:
                    ret_val, frame = video_capture.read()
                    print("FRAME:")
                    print(frame.shape, frame.dtype)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Check to see if the user closed the window
                    # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                    # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                    # if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    #     cv2.imshow(window_title, frame)

                    if not (frame is None):
                        return frame
                    # else:
                    #     break
                    # keyCode = cv2.waitKey(10) & 0xFF
                    # # Stop the program on the ESC key or 'q'
                    # if keyCode == 27 or keyCode == ord('q'):
                    #     break
            finally:
                video_capture.release()
                cv2.destroyAllWindows()
        else:
            print("Error: Unable to open camera")


    # if __name__ == "__main__":
    #     show_camera()

    def get_raw_img(self):
        f = self.read_camera()
        print("GOT IMAGE!!!!!!!!!!!")
        print("type:")
        print(type(f))
        return f

    def prepare_image(self, img, im_width, im_height):
        img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #img = img.astype(np.int32, copy=False)
        return img

    def get_cropped_img(self, optional_index=0):
        left = 200
        right = 800
        upper = 00
        lower = 830
        raw_img = self.get_raw_img()
        cropped_img = raw_img[upper:lower, left:right]
        cropped_img = self.prepare_image(cropped_img, 128, 128)
        #cv2.imshow("cropped", cropped_img)
        # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"rubik_face_test{str(optional_index)}.jpg", cropped_img)
        # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        # print("Wrote captured image as file. VIEW IT TO CHECK COLOR!!!")
        return cropped_img