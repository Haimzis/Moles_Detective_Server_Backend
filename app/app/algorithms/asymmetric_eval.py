from ..utils import utils
import cv2

THRESHOLD = 0.165


def asymmetric_eval(aligned_mask):
    """
    :param mask: segmentation mask of mole.
    :return: ratio between uncommon pixels and common pixels
             of the 2 parts that divided from the center both horizontal and vertical.
    """
    VAS = 0
    HAS = 0

    width_center, height_center = aligned_mask.shape[1] // 2, aligned_mask.shape[0] // 2

    # find 2 half for vertical and horizontal views.
    if aligned_mask.shape[1] % 2 == 0:
        left_half = aligned_mask[:, 0: width_center]
    else:
        left_half = aligned_mask[:, 0: width_center + 1]
    right_half = aligned_mask[:, width_center: aligned_mask.shape[1]]
    if aligned_mask.shape[0] % 2 == 0:
        upper_half = aligned_mask[0: height_center, :]
    else:
        upper_half = aligned_mask[0: height_center + 1, :]
    bottom_half = aligned_mask[height_center: aligned_mask.shape[0], :]

    overlapped_left_half = cv2.flip(left_half, 1)
    overlapped_bottom_half = cv2.flip(bottom_half, 0)

    HM_horizontal = hammoude_distance(right_half, overlapped_left_half)
    HM_vertical = hammoude_distance(upper_half, overlapped_bottom_half)

    if HM_horizontal > THRESHOLD:
        HAS = 1.0
    if HM_vertical > THRESHOLD:
        VAS = 1.0

    A = HAS + VAS
    score = min((((HM_horizontal + HM_vertical) / 2) / THRESHOLD) ** 2, 1.0)
    return score, A


def intersection(A_mask, B_mask):
    return cv2.bitwise_and(A_mask, B_mask)


def union(A_mask, B_mask):
    return cv2.bitwise_or(A_mask, B_mask)


def N(mask):
    return cv2.countNonZero(mask)


def hammoude_distance(A_mask, overlapped_B_mask):
    return (N(union(A_mask, overlapped_B_mask)) - N(intersection(A_mask, overlapped_B_mask))) / \
           N(union(A_mask, overlapped_B_mask))


if __name__ == '__main__':
    seg_mask = cv2.imread(
        '/home/haimzis/PycharmProjects/DL_training_preprocessing/Output/objects_extraction/classification_purpose/annotations/ISIC_0000019_downsampled.png',
        -1)
    # seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    seg_mask = utils.align_by_centroid(seg_mask)
    seg_mask = utils.cut_roi_from_mask(seg_mask, utils.find_object_coords(seg_mask))
    print(asymmetric_eval(seg_mask))


