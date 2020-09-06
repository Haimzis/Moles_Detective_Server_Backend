import cv2
import numpy as np


def color_eval(image, mask):
    img, mask = color_elimination(image, mask, 20)
    print('stam')


def color_elimination(image, mask, elimination_constant):
    gray_scale_object_img = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2GRAY).astype('uint8')
    color_high = np.array(np.max(gray_scale_object_img))
    color_low = np.array(np.subtract(np.max(gray_scale_object_img), elimination_constant), dtype=np.uint8)
    skin_mask = cv2.inRange(gray_scale_object_img, color_low, color_high)
    complement_skin_mask = cv2.bitwise_not(skin_mask)
    new_mask = cv2.bitwise_and(mask, mask, mask=complement_skin_mask)
    new_image = cv2.bitwise_and(image, image, mask=new_mask)
    return new_image, new_mask
