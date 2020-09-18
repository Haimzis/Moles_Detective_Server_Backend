import cv2
import numpy as np
from ..utils import params


def normalize_final_score(final_score):
    return min(1.0, (final_score / 6.0)**3)


def is_there_many_recognition(mask):
    first_recognition = find_object_coords(mask)
    ymin, ymax, xmin, xmax = first_recognition
    copy_mask = mask.copy()
    copy_mask[ymin: ymax + 1, xmin: xmax + 1] = 0
    second_recognition = find_object_coords(copy_mask)
    if second_recognition:
        return True
    return False


def verify_segmentation_mask(segmentation_output):
    for mask in segmentation_output:
        if not mask.any():
            raise Exception('mask is empty.')
        elif is_there_many_recognition(mask):
            raise Exception('many recognitions was found, this situation is not supported.')
        elif cv2.countNonZero(mask) < params.MIN_POSSIBLE_PIXELS_FOR_RECOGNITION:
            raise Exception('recognition contains too few pixels, less than {0}.'
                            .format(params.MIN_POSSIBLE_PIXELS_FOR_RECOGNITION))


def find_object_coords(object_mask, coords=None):  # crop_coords = [ymin, ymax, xmin, xmax]
    if not object_mask.any():
        return []

    if coords is None:
        min_y = 0
        max_y = object_mask.shape[0] - 1
        min_x = 0
        max_x = object_mask.shape[1] - 1
    else:
        min_y = coords[0]
        max_y = coords[1]
        min_x = coords[2]
        max_x = coords[3]

    # y coords
    while not object_mask[min_y, min_x:max_x].any():
        min_y += 1
    while not object_mask[max_y, min_x:max_x].any():
        max_y -= 1

    # x coords
    while not object_mask[min_y:max_y, min_x].any():
        min_x += 1
    while not object_mask[min_y:max_y, max_x].any():
        max_x -= 1

    return [min_y, max_y, min_x, max_x]


def cut_roi_from_mask(mask, coords):  # crop_coords = [ymin, ymax, xmin, xmax]
    return mask[coords[0]: coords[1], coords[2]: coords[3]]


def cut_roi_from_image(image, coords):  # crop_coords = [ymin, ymax, xmin, xmax]
    return image[coords[0]: coords[1], coords[2]: coords[3], :]


def rotate(mask, angle):
    """
    :param mask: segmentation mask
    :param angle: angle of rotation
    :return: rotated segmentation mask without information loss
    """
    height, width = mask.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mask, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def align_by_diameter(mask):
    """
    :param mask: segmentation mask of mole.
    :return: aligned segmentation mask of the mole.
    """
    alignment_res = mask.copy()
    best_rotation_size = mask.shape[0]
    for angle in range(0, 180):
        res = rotate(mask, angle)
        res = cut_roi_from_mask(res, find_object_coords(res))
        if best_rotation_size < res.shape[0]:
            best_rotation_size = res.shape[0]
            alignment_res = res
    for i in range(0, alignment_res.shape[1]):
        for j in range(0, alignment_res.shape[0]):
            if alignment_res[j, i].any():
                alignment_res[j, i] = 255
    return alignment_res


def align_by_centroid(mask):
    Yc, Xc = find_center_coords(mask)
    # calculate theta angle for centroid alignment
    m11 = 0
    m20 = 0
    m02 = 0
    for Xi in range(mask.shape[1]):
        for Yi in range(mask.shape[0]):
            if not mask[Yi, Xi].any():
                continue
            m11 += (Xi - Xc)**1 * (Yi - Yc)**1
            m20 += (Xi - Xc)**2 * (Yi - Yc)**0
            m02 += (Xi - Xc)**0 * (Yi - Yc)**2

    theta = 0.5*np.arctan(2*m11 / (m20 - m02))
    res = rotate(mask, np.degrees(theta))
    return res


def find_center_coords(mask_original):
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(mask_original, 127, 255, 0)
    # calculate moments of binary image
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    Xc = int(M["m10"] / M["m00"])
    Yc = int(M["m01"] / M["m00"])
    return Yc, Xc


def find_object_radius(center, mask_original_coords):
    radius = max(distance(center, (mask_original_coords[0], mask_original_coords[2])),
                 distance(center, (mask_original_coords[0], mask_original_coords[3])),
                 distance(center, (mask_original_coords[1], mask_original_coords[2])),
                 distance(center, (mask_original_coords[1], mask_original_coords[3])))
    return radius


def distance(coords1, coords2):
    return ((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2) ** 0.5


def reference_object_1ISL_recognition(reference_obj_image):
    ISL1_SIZE = 18  # mm
    roi = cv2.imread(reference_obj_image)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 2, roi.shape[0], param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(roi, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(roi, (i[0], i[1]), 2, (0, 0, 255), 3)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(roi, "coin", (0, 400), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.imshow('Detected coins', roi)
    # cv2.waitKey()

    return circles[0][0], ISL1_SIZE


if __name__ == '__main__':
    seg_mask = cv2.imread('/home/haimzis/1600431611138_0_mask.png',  -1)
    seg_mask = cut_roi_from_mask(seg_mask, find_object_coords(seg_mask))
    seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    print(find_center_coords(seg_mask))
    print(find_object_radius(seg_mask))
    ci = cv2.circle(seg_mask, find_center_coords(seg_mask), find_object_radius(seg_mask), (255, 0, 0), 1)
    print("stam")