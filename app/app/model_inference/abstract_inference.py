import os
import sys
from abc import ABC, abstractmethod
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
from ..utils import params
import cv2
import logging


class AbstractModelInference(ABC):
    """Abstract Class to load model and run inference."""
    def __init__(self, input_tensor_name: str, output_tensor_name: str, input_size: int, frozen_graph_name: str,
                 frozen_graph_path: str, image_preprocess_method: staticmethod):
        self.frozen_graph_name = frozen_graph_name
        self.input_size = input_size
        self.output_tensor_name = output_tensor_name
        self.input_tensor_name = input_tensor_name
        self.image_preprocess_func = image_preprocess_method
        self.sess = None
        self.load_graph(frozen_graph_path)

    def load_graph(self, frozen_graph_path):
        """Creates and loads pretrained deeplab model."""
        logging.info('loading DeepLab model...')
        graph = tf.Graph()
        graph_def = None
        with tf.gfile.FastGFile(frozen_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=graph)
        logging.info('model loaded successfully!')

    def run(self, image):
        """
            Runs inference on a single image.
        Args:
            image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        target_size = (self.input_size, self.input_size)
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        net_image = resized_image
        if self.image_preprocess_func:
            net_image = self.image_preprocess_func(resized_image)
            net_image = np.expand_dims(net_image, axis=-1)
        output = self.sess.run(
            self.output_tensor_name,
            feed_dict={self.input_tensor_name: [np.asarray(net_image)]})
        return resized_image, output

    @abstractmethod
    def run_visualization(self, image_path):
        pass

    @abstractmethod
    def quick_inference(self, image_path):
        pass
