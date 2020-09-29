import cv2
import numpy as np

colors_ranges = {  ## RGB
    'dark': [[1, 1, 1], [62, 52, 52]],
    'white': [[205, 205, 205], [255, 255, 255]],
    'red': [[150, 0, 0], [255, 51, 51]],
    'light_brown': [[150, 51, 1], [240, 150, 99]],
    'dark_brown': [[63, 0, 1], [149, 99, 99]],
    'blue_gray': [[0, 100, 125], [150, 125, 150]]
}


def decide_color(pixel):
    """
    :param pixel: rgb pixel
    :return: pixel color label
    """
    for color, color_range in colors_ranges.items():
        if is_in_range(color, pixel):
            return color
    return "else"


def color_eval(image, mask):
    """
    :param image: lesion image
    :param mask: lesion mask
    :return: float color evaluation, by the number of a unique colors appearance.
    """
    lesion_only = cv2.bitwise_and(image, image, mask=mask)
    colors_counter = {color: 0 for color in colors_ranges}
    colors_counter['else'] = 0
    C = 0
    skin_color = skin_color_assumption(image, mask)
    num_of_pixels = cv2.countNonZero(cv2.cvtColor(lesion_only, cv2.COLOR_BGR2GRAY))
    for i in range(lesion_only.shape[0]):
        for j in range(lesion_only.shape[1]):
            if not lesion_only[i, j].any():
                continue
            colors_counter[decide_color(lesion_only[i, j, :])] += 1
    if is_in_range('white', skin_color):
        colors_counter.pop('white')
    for color, color_appearance in colors_counter.items():
        if color_appearance > num_of_pixels / 100 \
                and color != 'else':
            C += 1
    color_score = C / 5
    return color_score, C


def skin_color_assumption(image, mask):
    """
    :param image: lesion img
    :param mask: lesion mask
    :return: estimated skin color range, after average color calculation
    """
    skin_only = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    skin_non_zero_pixels_amount = cv2.countNonZero(cv2.cvtColor(skin_only, cv2.COLOR_BGR2GRAY))
    average_skin_color = (np.array(cv2.sumElems(skin_only))[:3] // skin_non_zero_pixels_amount).astype('uint8')
    return average_skin_color


def color_mask_extraction(image, color_borders):
    """
    :param image: lesion img
    :param color_borders: range of rgb colors
    :return: mask that contains the pixels that included in the range of color_borders
    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for color_border in color_borders:
        hsv_color1 = cv2.cvtColor(np.uint8([[color_border[0]]]), cv2.COLOR_BGR2HSV)
        hsv_color2 = cv2.cvtColor(np.uint8([[color_border[1]]]), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
        color_mask = cv2.bitwise_or(color_mask, mask)
    return color_mask


def is_in_range(color, pixel):
    """
    :param color: color label
    :param pixel: rgb pixel
    :return: checks if the given pixel has the given color label.
    """
    color_range = colors_ranges[color]
    rmin, gmin, bmin = color_range[0]
    rmax, gmax, bmax = color_range[1]
    b, g, r = list(pixel)
    if rmax >= r >= rmin \
            and gmax >= g >= gmin \
            and bmax >= b >= bmin:
        return True
    return False


if __name__ == '__main__':
    image = cv2.imread('/home/haimzis/1600431611138_0.png', -1)
    mask = cv2.cvtColor(cv2.imread('/home/haimzis/1600431611138_0_mask.png', -1), cv2.COLOR_BGR2GRAY)
    color_eval(image, mask)
