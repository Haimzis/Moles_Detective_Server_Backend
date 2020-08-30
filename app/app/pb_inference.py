import os
import sys
from io import BytesIO
import tarfile
import tempfile
import random

from six.moves import urllib
import time
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from .utils import params
import cv2


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = params.INPUT_SIZE
    FROZEN_GRAPH_NAME = 'my_frozen_inference_graph'

    def __init__(self, MY_GRAPH_PATH):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        with tf.gfile.FastGFile(MY_GRAPH_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
        target_size = (self.INPUT_SIZE, self.INPUT_SIZE)
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        net_image = resized_image
        # if params.HZ_preprocess_activate:
        #     net_image = params.image_preprocess_func(resized_image)
        #     net_image = np.expand_dims(net_image, axis=-1)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(net_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def run_visualization(image):
    """Inferences DeepLab model and visualizes result."""
    # for image in images:
    try:
        with tf.gfile.FastGFile(image, 'rb') as f:
            jpeg_str = f.read()
            original_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image')
        return

    # print('running deeplab on image {0}'.format(image))
    resized_im, seg_map = MODEL.run(original_im)
    seg_map = seg_map.astype(np.uint8) * 255
    resized_im = np.array(resized_im, dtype=np.uint8)
    resized_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2RGB)
    overlay_image = cv2.addWeighted(resized_im, 0.8, cv2.merge((seg_map * 0, seg_map, seg_map * 0)), 0.2, 0)

    return resized_im, seg_map, overlay_image.astype(np.uint8)


MODEL = None
FULL_LABEL_MAP = None
FULL_COLOR_MAP = None
MY_GRAPH_PATH = None
LABEL_NAMES = None


def init_inference():
    global MODEL
    global FULL_LABEL_MAP
    global FULL_COLOR_MAP
    global MY_GRAPH_PATH
    global LABEL_NAMES

    ### Loading of NN ###
    LABEL_NAMES = np.asarray([
        'background', 'Mole'
    ])

    MY_GRAPH_PATH = '/app/files/Models/MobileNet_V3_large_ISIC_ver1.pb'
    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    print('loading DeepLab model...')
    MODEL = DeepLabModel(MY_GRAPH_PATH)
    print('model loaded successfully!')


def quick_inference(img_input):
    return run_visualization(img_input)
