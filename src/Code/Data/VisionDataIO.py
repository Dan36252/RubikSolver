import torch
from torch import nn
from PIL import Image
import cv2
import numpy as np
import re
from pathlib import Path
import time, random
#from VisionAlgo2 import get_img_sectors_3x3

def rotate_3x3_grid(arr):
    # Clockwise 90 degrees
    new_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_arr[0] = arr[6]
    new_arr[1] = arr[3]
    new_arr[2] = arr[0]
    new_arr[3] = arr[7]
    new_arr[4] = arr[4]
    new_arr[5] = arr[1]
    new_arr[6] = arr[8]
    new_arr[7] = arr[5]
    new_arr[8] = arr[2]
    #print(arr)
    #print(new_arr)
    return new_arr

def reshape_channels_first(arr):
    output = np.zeros((3, len(arr), len(arr[0])))
    for r in range(len(arr)):
        for c in range(len(arr[r])):
            for channel in range(len(arr[r][c])):
                output[channel][r][c] = arr[r][c][channel]
    return output

def remove_val_channel(arr):
    output = np.zeros((len(arr), len(arr[0]), 2))
    for r in range(len(arr)):
        for c in range(len(arr[r])):
            output[r][c][0] = arr[r][c][0]
            output[r][c][1] = arr[r][c][1]
    return output

def two_channel_hsv_to_norm(arr):
    output = np.zeros((len(arr), len(arr[0]), 3))
    for r in range(len(arr)):
        for c in range(len(arr[r])):
            output[r][c][0] = int(arr[r][c][0]/2)
            output[r][c][1] = arr[r][c][1]
            output[r][c][2] = arr[r][c][2]
    return np.array(output, dtype=np.uint8)


# Primary Data Loading Method
def load_data(im_width=24, im_height=24, images_path="ALLFaceData", labels_path="ALLFaceDataLabels.txt", hsv=False):
    # First read existing labeled data and get the rotated versions of that data
    X_train, Y_train, X_test, Y_test = get_data(im_width, im_height, images_path, labels_path, hsv)
    # print(f"Data length: X - {len(X)}, y - {len(y)}")
    # print("X:")
    # print(X)
    # print("y:")
    # print(y)

    # Verify correct labels (temp)
    # if hsv == False:
    #     i = 0
    #     while i < len(X)-1:
    #         i += 1
    #         img = X[i]
    #         print("img:")
    #         print(img)
    #         #img = two_channel_hsv_to_norm(img)
    #         print(f"img shape: {img.shape}")
    #         img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
    #         cv2.imshow("hi", img)
    #         print(f"Label: {y[i]}")
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()


    # Then generate extra data
    # X2, y2 = generate_data(10000, red=[175, 45, 45], yellow=[200, 70, 70], green=[100, 200, 90], white=[190, 200, 200], orange=[220, 100, 40], blue=[25, 95, 150], im_width=im_width, im_height=im_height)
    # X = np.concat((np.array(X, dtype=np.uint8), np.array(X2, dtype=np.uint8)))
    # y = np.concat((np.array(y, dtype=np.uint8), np.array(y2, dtype=np.uint8)))
    print("Data loaded!!!")
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    return X_train, Y_train, X_test, Y_test

def generate_shaky_images(img, crop_decrease_size):
    # Number of images generated = crop_decrease_size ^ 2
    orig_size_x = len(img)
    orig_size_y = len(img[0])
    size_x = orig_size_x-crop_decrease_size
    size_y = orig_size_y-crop_decrease_size
    output = []
    for x in range(crop_decrease_size):
        for y in range(crop_decrease_size):
            cropped = img.copy()
            cropped = cropped[y:y+size_y, x:x+size_x]
            cropped = cv2.resize(cropped, (orig_size_x, orig_size_y))
            # Temp test
            # cropped_img = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR_FULL)
            # cv2.imshow("cropped", cropped_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            output.append(cropped)
    return output

def generate_light_variations(imgs_list, n=7, hsv=True):
    output = []
    for img in imgs_list:
        for i in range(n):
            m = img.copy()
            test = m.copy()
            for r in range(len(m)):
                for c in range(len(m[r])):
                    if hsv:
                        # Assuming Value Channel was already removed.
                        change_radius = 0.3
                        change_factor = 1 + (change_radius*2*(i/(n-1)) - change_radius)
                        m[r][c][1] = min(max(m[r][c][1] * change_factor, 0), 255)
                        # test[r][c][1] = m[r][c][1]
                    else:
                        m[r][c][0] = min(m[r][c][0] * (0.8 + (0.4 * i / n)), 255)
                        m[r][c][1] = min(m[r][c][1] * (0.8 + (0.4 * i / n)), 255)
                        m[r][c][2] = min(m[r][c][2] * (0.8 + (0.4 * i / n)), 255)
            output.append(m)
            # Temp test
            # test = np.concatenate((test, np.ones((m.shape[0], m.shape[1], 1))*70), axis=2, dtype=np.float32)
            # test = cv2.cvtColor(test, cv2.COLOR_HSV2BGR_FULL)
            # print(test.shape)
            # print(test)
            #
            # cv2.imshow("light", m)
            # cv2.imwrite("TESTING_LIGHT.jpg", test)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    return output


def get_data(im_width=24, im_height=24, images_path="ALLFaceData", labels_path="ALLFaceDataLabels.txt", hsv=True):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    test_sample_every = 30
    crop_decrease_size = 3
    num_light_shades = 7

    i = 0
    images_path = Path(images_path)
    for raw_img in sorted(images_path.iterdir(), key=lambda item: item.name):
        if re.search(".jpg", raw_img.name) is None: continue
        print(raw_img.name)
        img = cv2.imread(images_path.name + "/" + raw_img.name)
        img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
        # print(img)
        # cv2.imshow("test1", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if hsv:
            do_nothing = True
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
            # for r in range(len(img)):
            #     for c in range(len(img[r])):
            #         img[r][c][2] = 70
            # img2 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)

            # print(img2)
            # cv2.imshow("test2", img2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #img = remove_val_channel(img)
        jitters = generate_shaky_images(img, crop_decrease_size)
        lights = generate_light_variations(jitters, num_light_shades, hsv)
        for l in lights:
            imgR0 = l
            imgR1 = cv2.rotate(imgR0, cv2.ROTATE_90_CLOCKWISE)
            imgR2 = cv2.rotate(imgR1, cv2.ROTATE_90_CLOCKWISE)
            imgR3 = cv2.rotate(imgR2, cv2.ROTATE_90_CLOCKWISE)

            if i % test_sample_every == 0:
                X_test.append(reshape_channels_first(imgR0))
                X_test.append(reshape_channels_first(imgR1))  # .flatten())
                X_test.append(reshape_channels_first(imgR2))  # .flatten())
                X_test.append(reshape_channels_first(imgR3))  # .flatten())
            else:
                X_train.append(reshape_channels_first(imgR0))
                X_train.append(reshape_channels_first(imgR1))#.flatten())
                X_train.append(reshape_channels_first(imgR2))#.flatten())
                X_train.append(reshape_channels_first(imgR3))#.flatten())

        i += 1

    l = 0
    with open(labels_path, "r") as file:
        for line in file.readlines():
            for i in range(crop_decrease_size*crop_decrease_size):
                for j in range(num_light_shades):
                    label = list(line.strip())
                    # print(label)
                    labelR1 = rotate_3x3_grid(label)
                    labelR2 = rotate_3x3_grid(labelR1)
                    labelR3 = rotate_3x3_grid(labelR2)

                    if l % test_sample_every == 0:
                        Y_test.append(label)
                        Y_test.append(labelR1)
                        Y_test.append(labelR2)
                        Y_test.append(labelR3)
                    else:
                        Y_train.append(label)
                        Y_train.append(labelR1)
                        Y_train.append(labelR2)
                        Y_train.append(labelR3)
            l += 1

    return np.array(X_train, dtype=np.uint8), np.array(Y_train, dtype=np.uint8), np.array(X_test, dtype=np.uint8), np.array(Y_test, dtype=np.uint8)

def generate_data(n, red, yellow, green, white, orange, blue, im_width=24, im_height=24):
    X = []
    y = []
    colors = [red, yellow, green, white, orange, blue]

    for i in range(n):

        # Determine all 9 face colors
        face_colors = []
        code = []
        for r in range(3):
            for c in range(3):
                color_code = random.randint(0, 5)
                color = colors[color_code].copy()
                lighting = random.randint(-20, 20)
                for i in range(len(color)):
                    color[i] = min(max(color[i]+lighting, 0), 255)
                code.append(color_code)
                face_colors.append(color)

        # Create full image
        img = np.zeros((3, im_height, im_width))
        for r in range(im_height):
            for c in range(im_width):
                color_col = min(int(c/(im_width/3)), 2)
                color_row = min(int(r/(im_height/3)), 2)
                color_index = color_row*3+color_col
                color = face_colors[color_index]
                img[0][r][c] = max(min(color[0] + random.randint(-10, 10), 255), 0)
                img[1][r][c] = max(min(color[1] + random.randint(-10, 10), 255), 0)
                img[2][r][c] = max(min(color[2] + random.randint(-10, 10), 255), 0)

        # print("IMG:")
        # print(img)
        # print("Code:")
        # print(code)
        # print(np.array(img).shape)
        # tmp = np.array(img, dtype=np.uint8).reshape((24, 24, 3))
        # tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        # print("TEMP:")
        # print(tmp)
        # cv2.imshow("test", tmp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        X.append(img)
        y.append(code)

    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.uint8)

def create_old_dataset_txt(images_path, labels_path, im_width=24, im_height=24):
    rotated_faces = [] # Used to generate more data from existing images
    rotated_labels = []

    # Read original rubik's cube face images folder and create data text files
    with open("RubikFaceData.txt", "w") as file:
        images_path = Path(images_path)
        sorted_images = sorted(images_path.iterdir(), key=lambda item: item.name)
        for raw_img in sorted_images:
            if re.search(".jpg", raw_img.name) is None: continue
            img = cv2.imread(images_path.name + "/" + raw_img.name)
            img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
            imgR1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            imgR2 = cv2.rotate(imgR1, cv2.ROTATE_90_CLOCKWISE)
            imgR3 = cv2.rotate(imgR2, cv2.ROTATE_90_CLOCKWISE)
            # cv2.imshow(raw_img.name, imgR1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img.shape)
            features = img.flatten()
            featuresR1 = imgR1.flatten()
            featuresR2 = imgR2.flatten()
            featuresR3 = imgR3.flatten()
            features_str = ""
            features_str_R1 = ""
            features_str_R2 = ""
            features_str_R3 = ""
            for i in range(len(features)):
                features_str = features_str + " " + str(features[i])
                features_str_R1 = features_str_R1 + " " + str(featuresR1[i])
                features_str_R2 = features_str_R2 + " " + str(featuresR2[i])
                features_str_R3 = features_str_R3 + " " + str(featuresR3[i])
            file.write(features_str + "\n")
            rotated_faces.append(features_str_R1)
            rotated_faces.append(features_str_R2)
            rotated_faces.append(features_str_R3)

    with open(labels_path, "r") as file:
        for line in file.readlines():
            label_array = list(line.strip())
            label_R1 = rotate_3x3_grid(label_array)
            label_R2 = rotate_3x3_grid(label_R1)
            label_R3 = rotate_3x3_grid(label_R2)
            rotated_labels.append("".join(label_R1))
            rotated_labels.append("".join(label_R2))
            rotated_labels.append("".join(label_R3))

    with open("RotatedFaceData.txt", "w") as file:
        for img in rotated_faces:
            file.write(img+"\n")

    with open("RotatedFaceLabels.txt", "w") as file:
        for code in rotated_labels:
            file.write(code+"\n")

def get_old_dataset(features_path, labels_path):
    X = []
    y = []

    with open(features_path, "r") as file:
        for line in file.readlines():
            X.append(line.strip().split())

    with open(labels_path, "r") as file:
        for i, line in enumerate(file.readlines()):
            y.append(list(line.strip()))

    with open("RotatedFaceData.txt", "r") as file:
        for line in file.readlines():
            X.append(line.strip().split())

    with open("RotatedFaceLabels.txt", "r") as file:
        for i, line in enumerate(file.readlines()):
            y.append(list(line.strip()))

    return np.array(X, dtype=np.int16), np.array(y, dtype=np.int16)

def generate_old_data(n, red, yellow, green, white, orange, blue, im_width=24, im_height=24):
    X = []
    y = []
    colors = [red, yellow, green, white, orange, blue]

    for i in range(n):

        # Determine all 9 face colors
        face_colors = []
        code = []
        for r in range(3):
            for c in range(3):
                color_code = random.randint(0, 5)
                color = colors[color_code].copy()
                lighting = random.randint(-30, 30)
                for i in range(len(color)):
                    color[i] = min(max(color[i]+lighting, 0), 255)
                code.append(color_code)
                face_colors.append(color)

        # Create full image
        img = []
        for r in range(IM_HEIGHT):
            for c in range(IM_WIDTH):
                color_col = min(int(c/(im_width/3)), 2)
                color_row = min(int(r/(im_height/3)), 2)
                color_index = color_row*3+color_col
                color = face_colors[color_index]
                img.append(max(min(color[0] + random.randint(-10, 10), 255), 0))
                img.append(max(min(color[1] + random.randint(-10, 10), 255), 0))
                img.append(max(min(color[2] + random.randint(-10, 10), 255), 0))

        # print("IMG:")
        # print(img)
        # print("Code:")
        # print(code)
        # print(np.array(img).shape)
        # tmp = np.array(img, dtype=np.uint8).reshape((24, 24, 3))
        # tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        # print("TEMP:")
        # print(tmp)
        # cv2.imshow("test", tmp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        X.append(img)
        y.append(code)

    return np.array(X, dtype=np.int16), np.array(y, dtype=np.int16)