from Camera import Camera
import cv2, time, math
import numpy as np

# Main method for using the camera to get the colors on the currently visible cube face
def read_cube_face(optional_img):
    img = None
    if optional_img is None:
        cam = Camera()
        img = cam.get_cropped_img()
    else:
        img = optional_img

    img = prepare_image(img, 48, 48)

    sectors = get_img_sectors_3x3(img)
    colors = get_avg_sector_colors(sectors)
    return colors

def get_avg_sector_colors(sectors):
    # HSV colors: red, yellow, green, white, orange, blue. (hue, saturation) [NO VALUE]
    # MAKE SURE TO IGNORE HUE WHEN CALCULATING DISTANCE FROM WHITE (in hue vs saturation space)
    hsv_cube_colors = [[0, 195], [40, 179], [85, 153], [10000, 4], [15, 204], [110, 207]]
    colors = []

    # In each sector, find the color that its pixels are closest to.
    for sector in sectors:
        color_distances = [0, 0, 0, 0, 0, 0] # This sector's pixels' total distances from each HSV color
        for r in range(len(sector)):
            for c in range(len(sector[r])):
                for clr in range(len(hsv_cube_colors)):
                    pixel = sector[r][c]
                    refcol = hsv_cube_colors[clr]
                    print(f"Pixel = {pixel}, Reference Col = {refcol}")
                    hue_dif = (pixel[0] - refcol[0])**2 if not (clr == 0 and abs(refcol[0] - pixel[0]) >= 90) else (pixel[0] - 180)**2
                    sat_dif = (pixel[1] - refcol[1])**2
                    dist = math.sqrt(hue_dif + (sat_dif/50)) if clr != 3 else math.sqrt(sat_dif)
                    color_distances[clr] += dist
        # Now get minimum color distance to identify which color this sector most likely is
        min_dist_index = 0
        min_dist = color_distances[0]
        for i in range(len(color_distances)):
            if color_distances[i] < min_dist:
                min_dist_index = i
                min_dist = color_distances[i]

        print("Color Distances:")
        print(color_distances)
        cv2.imshow("sector", cv2.cvtColor(np.astype(sector, np.uint8), cv2.COLOR_HSV2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        colors.append(min_dist_index)

    return colors

def prepare_image(img, im_width, im_height):
    img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = np.astype(img, np.int32, copy=False)
    return img

def get_img_sectors_3x3(img):
    padding = 3
    sectors = []
    sector_height = int(len(img)/3)
    sector_width = int(len(img[0])/3)
    #print(f"Supposed sector dimensions: {sector_height} x {sector_width}")
    for r in range(3):
        for c in range(3):
            sector_start_x = c * sector_width + padding
            sector_end_x = sector_start_x + sector_width - (2*padding) if c < 2 else len(img[0]) - (2*padding)
            sector_start_y = r * sector_height + padding
            sector_end_y = sector_start_y + sector_height - (2*padding) if r < 2 else len(img) - (2*padding)
            #print(f"Start X: {sector_start_x}, End X: {sector_end_x}; Start Y: {sector_start_y}, End Y: {sector_end_y}")
            sectors.append(img[sector_start_y:sector_end_y, sector_start_x:sector_end_x])
    print(f"Sectors length: {len(sectors)} x {len(sectors[0])} x {len(sectors[0][0])}")
    return sectors


# TEST
path = "RubikFaceData/04.jpg"
path1 = "RubikFaceData/07.jpg"
path2 = "RubikFaceData/15.jpg"
img = cv2.imread(path)
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
print("VISION TEST:")
print(f"{path} --> {read_cube_face(img)}")
print(f"{path1} --> {read_cube_face(img1)}")
print(f"{path2} --> {read_cube_face(img2)}")
