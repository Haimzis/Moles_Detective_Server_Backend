import logging
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from ..model_inference.abstract_inference import AbstractModelInference
import cv2


class ClassificationModelInference(AbstractModelInference):
    """Class to load deeplab model and run inference."""

    def __init__(self, input_tensor_name: str, output_tensor_name: str, input_size: int, frozen_graph_name: str,
                 frozen_graph_path: str, image_preprocess_method: staticmethod, batch_size: int):
        super().__init__(input_tensor_name, output_tensor_name, input_size, frozen_graph_name, frozen_graph_path,
                         image_preprocess_method)
        self.batch_size = batch_size
        self.label_names = {
            0: 'AK',
            1: 'BCC',
            2: 'BKL',
            3: 'DF',
            4: 'MEL',
            5: 'NV',
            6: 'SCC',
            7: 'UNK',
            8: 'VASC'
        }

    def run_visualization(self, img_input):
        pass

    def quick_inference(self, img_input):
        """Inferences DeepLab model and visualizes result."""
        # for image in images:
        try:
            with tf.gfile.FastGFile(img_input, 'rb') as f:
                jpeg_str = f.read()
                original_im = Image.open(BytesIO(jpeg_str))
        except IOError:
            print('Cannot retrieve image')
            return

        logging.info(
            'running {model_name} segmentation model on image {image}'.format(model_name=self.frozen_graph_name,
                                                                              image=img_input))
        resized_im, classification_output = self.run(original_im, self.batch_size, np.float32)
        classification_output = classification_output.sum(axis=0) / classification_output.shape[0]
        label_index = np.argmax(classification_output)
        return cv2.cvtColor(np.array(resized_im, dtype=np.uint8), cv2.COLOR_RGB2BGR), (self.label_names[label_index], classification_output[label_index])
