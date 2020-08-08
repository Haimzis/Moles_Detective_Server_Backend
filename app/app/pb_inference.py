from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import params
import cv2


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = params.INPUT_SIZE
    FROZEN_GRAPH_NAME = 'my_frozen_inference_graph'

    def __init__(self):
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
        if params.HZ_preprocess_activate:
            net_image = params.image_preprocess_func(resized_image)
            net_image = np.expand_dims(net_image, axis=-1)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(net_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def run_model(image):
    """Inferences DeepLab model and visualizes result."""
    try:
        with tf.gfile.FastGFile(image, 'rb') as f:
            jpeg_str = f.read()
            original_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image')
        return

    print('running deeplab on image {0}'.format(image))
    resized_im, seg_map = MODEL.run(original_im)
    seg_map = seg_map.astype(np.uint8) * 255
    resized_im = np.array(resized_im, dtype=np.uint8)
    resized_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2RGB)
    overlay_image = cv2.addWeighted(resized_im, 0.8, cv2.merge((seg_map*0, seg_map, seg_map*0)), 0.2, 0)
    return resized_im, seg_map, overlay_image.astype(np.uint8)


### Loading of NN ###
LABEL_NAMES = np.asarray([
    'background', 'Mole'
])

MY_GRAPH_PATH = '../files/Models/MobileNet_V3_large_ISIC_ver1.pb'
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)

print('loading DeepLab model...')
MODEL = DeepLabModel()
print('model loaded successfully!')


def quick_inference(img_input_path):
    return run_model(img_input_path)
