import numpy as np
import numpy.core.multiarray
import cv2
from ..utils import utils


# PPM = Pixels Per Metric
# formula for calculating the lesion size:
# lesion real size = lesion_pixels_size / PPM
def calculate_pixels_size(mask):
    aligned_mask = utils.align(mask)
    mask_after_cut = utils.cut_roi_from_mask(aligned_mask, utils.find_object_coords(aligned_mask))
    return mask_after_cut.shape[:2]


def calculate_real_size(full_image, mask):
    reference_obj_circle, real_object_size = utils.reference_object_1ISL_recognition(full_image)
    lesion_height, lesion_width = calculate_pixels_size(mask)
    PPM = real_object_size / real_object_size
    skin_lesion_real_width, skin_lesion_real_height = lesion_height / PPM, lesion_width / PPM
    return max(skin_lesion_real_height, skin_lesion_real_width)  # mm


def size_eval(full_image, mask):
    skin_lesion_real_size = calculate_real_size(full_image, mask)
    return min((skin_lesion_real_size / 20) ** 1.5, 1.0)
