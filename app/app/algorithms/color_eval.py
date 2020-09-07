import cv2
import numpy as np
from app.app.utils import utils
# colors = {
#     'bright_blue': [[[109, 70, 132], [149, 90, 212]], [[158, 28, 212], [178, 48, 242]],
#                     [[127, 49, 154], [147, 69, 184]], [[142, 110, 110], [162, 130, 140]]],
#     'blue': [[[115, 95, 64], [170, 135, 144]],
#              [[144, 100, 43], [164, 120, 73]], [[120, 73, 56], [140, 93, 86]]],
#     'gray': [[[-9, 78, 125], [11, 98, 165]], [[-10, 64, 89], [10, 84, 119]], [[151, 122, 68], [171, 142, 98]],
#              [[160, 87, 129], [180, 107, 159]], [[-4, 48, 100], [16, 68, 130]],
#              [[-4, 98, 72], [16, 118, 102]]],
#     'red': [[[170, 100, 100], [180, 255, 255]],
#             [[-8, 35, 167] , [12, 55, 207]], [[169, 43, 167] , [189, 63, 207]],
#             [[166, 87, 143], [186, 107, 173]]],
#     'bright_brown': [[[-33, 189, 24], [47, 249, 144]], [[-3, 99, 128], [17, 119, 158]],
#             [[6, 165, 97], [26, 185, 127]], [[1, 210, 162], [21, 230, 192]],
#                      [[0, 202, 155], [20, 222, 185]], [[165, 74, 61], [185, 94, 91]],
#                      [[-10, 174, 135], [10, 194, 165]], [[-10, 187, 112], [10, 207, 142]],
#                      [[-8, 184, 148], [12, 204, 178]], [[-8, 175, 153], [12, 195, 183]]],
#     'brown': [[[-3, 88, 50], [17, 108, 80]], [[169, 68, 50], [189, 88, 80]],
#               [[-4, 117, 61], [16, 137, 91]], [[0, 167, 67], [20, 187, 97]],
#               [[168, 168, 28], [188, 188, 58]], [[-1, 237, 107], [19, 257, 137]],
#               [[161, 192, 52], [181, 212, 82]], [[165, 203, 71], [185, 223, 101]]],
#     'dark': [[[-29, 98, -44], [51, 158, 76]], [[1, 1, 0], [10, 10, 27]],
#             [[-10, 80, 2], [10, 100, 32]], [[6, 113, 16], [26, 133, 46]],
#             [[10, 33, 3], [30, 53, 33]], [[1, 83, 7], [21, 103, 37]],
#              [[-8, 163, 38], [12, 183, 68]]]
# }
colors_ranges = {  ## RGB
    'dark': [[1, 1, 1], [62, 52, 52]],
    'white': [[205, 205, 205], [255, 255, 255]],
    'red': [[150, 0, 0], [255, 51, 51]],
    'light_brown': [[150, 51, 1], [240, 150, 99]],
    'dark_brown': [[63, 0, 1], [149, 99, 99]],
    'blue_gray': [[0, 100, 125], [150, 125, 150]]
}


def decide_color(pixel):
    for color, color_range in colors_ranges.items():
        if is_in_range(color, pixel):
            return color
    return "else"


def color_eval(image, mask):
    lesion_only = cv2.bitwise_and(image, image, mask=mask)
    colors_counter = {color: 0 for color in colors_ranges}
    colors_counter['else'] = 0
    C = 0
    skin_color = skin_color_assumption(image, mask)
    if is_in_range('white', skin_color):
        colors_counter.pop('white')
    num_of_pixels = cv2.countNonZero(cv2.cvtColor(lesion_only, cv2.COLOR_BGR2GRAY))
    for i in range(lesion_only.shape[0]):
        for j in range(lesion_only.shape[1]):
            if not lesion_only[i, j].any():
                continue
            colors_counter[decide_color(lesion_only[i, j, :])] += 1
    for color, color_appearance in colors_counter.items():
        if color_appearance > num_of_pixels / 100 \
                and color != 'else':
            C += 1
    color_score = C / 6
    return color_score, C


def skin_color_assumption(image, mask):
    skin_only = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    skin_non_zero_pixels_amount = cv2.countNonZero(cv2.cvtColor(skin_only, cv2.COLOR_BGR2GRAY))
    average_skin_color = (np.array(cv2.sumElems(skin_only))[:3] // skin_non_zero_pixels_amount).astype('uint8')
    return average_skin_color


def color_mask_extraction(image, color_borders):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for color_border in color_borders:
        hsv_color1 = cv2.cvtColor(np.uint8([[color_border[0]]]), cv2.COLOR_BGR2HSV)
        hsv_color2 = cv2.cvtColor(np.uint8([[color_border[1]]]), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
        color_mask = cv2.bitwise_or(color_mask, mask)
    return color_mask


def is_in_range(color, pixel):
    color_range = colors_ranges[color]
    rmin, gmin, bmin = color_range[0]
    rmax, gmax, bmax = color_range[1]
    b, g, r = list(pixel)
    if rmax >= r >= rmin \
            and gmax >= g >= gmin \
            and bmax >= b >= bmin:
        return True
    return False