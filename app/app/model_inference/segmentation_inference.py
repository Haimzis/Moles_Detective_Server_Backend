import logging
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from app.app.model_inference.abstract_inference import AbstractModelInference
import cv2


class SegmentationModelInference(AbstractModelInference):
    """Class to load deeplab model and run inference."""

    def __init__(self, input_tensor_name: str, output_tensor_name: str, input_size: int, frozen_graph_name: str,
                 frozen_graph_path: str, image_preprocess_method: staticmethod):
        super().__init__(input_tensor_name, output_tensor_name, input_size, frozen_graph_name, frozen_graph_path,
                         image_preprocess_method)
        self.label_names = np.asarray([
            'background', 'Mole'
        ])

    def run_visualization(self, img_input):
        resized_im, seg_map = self.quick_inference(img_input)
        resized_im = np.array(resized_im, dtype=np.uint8)
        resized_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2RGB)
        overlay_image = cv2.addWeighted(resized_im, 0.8, cv2.merge((seg_map * 0, seg_map, seg_map * 0)), 0.2, 0)

        return resized_im, seg_map, overlay_image.astype(np.uint8)

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
        resized_im, seg_map = self.run(original_im)
        seg_map = seg_map.astype(np.uint8) * 255
        return resized_im, seg_map
