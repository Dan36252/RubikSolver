from PIL import Image

def get_raw_img():
    print("NOT IMPLEMENTED get_raw_img()")

def get_cropped_img():
    left = 100
    upper = 50
    right = 200
    lower = 150
    raw_img = get_raw_img()
    cropped_img = raw_img.crop((left, upper, right, lower))
    return cropped_img