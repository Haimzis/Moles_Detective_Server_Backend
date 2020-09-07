import sys

from flask import Flask, jsonify, request

from app.app.algorithms.asymmetric_eval import asymmetric_eval
from app.app.algorithms.border_eval import border_eval
from app.app.algorithms.classification_eval import classification_eval
from app.app.algorithms.color_eval import color_eval
from app.app.algorithms.final_evaluation import final_evaluation
from app.app.algorithms.size_eval_by_reference_obj import size_eval
from app.app.classes.Mole import Mole
from app.app.model_inference.classification_inference import ClassificationModelInference
from app.app.model_inference.segmentation_inference import SegmentationModelInference
from app.app.utils import log, params
from app.app.utils.upload_image import upload_file
from app.app.utils.utils import find_object_coords, find_center_coords, find_object_radius, cut_roi_from_mask
from app.app.utils.params import net_params

# app = Flask(__name__)
#
#
# @app.route("/")
# def hello():
#     print("Hello World", file=sys.stderr)
#     return "Hello"


# @app.route("/api/analyze", methods=['POST'])
def analyze():
    # image_path = upload_file(request)
    image_path = '/home/haimzis/PycharmProjects/yearly_project_flask/app/files/TestInputs/ISIC_0032206.jpg'
    # dpi = request.args['dpi']
    dpi = 411
    log.writeToLogs("Starting to check a new image: " + image_path)
    # separated_masks = prediction.separate_objects_from_mask(mask) TODO: in the future we will separate more than one mask
    # classification #
    classification_inference_instance = ClassificationModelInference(net_params.classification.input_tensor_name,
                                                                     net_params.classification.output_tensor_name,
                                                                     params.INPUT_SIZE,
                                                                     net_params.classification.frozen_model_name,
                                                                     net_params.classification.frozen_model, None,
                                                                     net_params.classification.batch_size)
    resized_image, classification_output = classification_inference_instance.quick_inference(image_path)
    # segmentation #
    segmentation_inference_instance = SegmentationModelInference(net_params.segmentation.input_tensor_name,
                                                                 net_params.segmentation.output_tensor_name,
                                                                 params.INPUT_SIZE,
                                                                 net_params.segmentation.frozen_model_name,
                                                                 net_params.segmentation.frozen_model, None)
    resized_image, segmentation_output = segmentation_inference_instance.quick_inference(image_path)

    # evaluation
    classification_score = classification_eval(classification_output)

    moles_analyze_results = []
    for index, separated_mask in enumerate(segmentation_output):
        separated_mask = cut_roi_from_mask(separated_mask, find_object_coords(separated_mask))
        border_score = border_eval(separated_mask)
        asymmetric_score = asymmetric_eval(separated_mask)
        size_score = size_eval('/home/haimzis/Desktop/index.jpeg', separated_mask)
        color_score = color_eval(resized_image, separated_mask)
        mole_coordinate = find_object_coords(separated_mask)
        mole_center = find_center_coords(mole_coordinate)
        mole_radius = find_object_radius(mole_coordinate)
        final_score = final_evaluation(border_score, size_score, asymmetric_score, color_score, classification_score)
        moles_analyze_results.append(
            Mole(asymmetric_score, size_score, border_score, color_score, final_score, classification_score,
                 mole_coordinate, mole_center, mole_radius))
    return jsonify({'results': moles_analyze_results.toJSON()})


if __name__ == "__main__":
    # Only for debugging while developing
    # app.run(host="0.0.0.0", debug=True, port=80)
    analyze()
