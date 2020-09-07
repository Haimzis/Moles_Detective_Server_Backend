import sys

from flask import Flask, jsonify, request
import json
from .algorithms.asymmetric_eval import asymmetric_eval
from .algorithms.border_eval import border_eval
from .algorithms.classification_eval import classification_eval
from .algorithms.color_eval import color_eval
from .algorithms.final_evaluation import final_evaluation
from .algorithms.size_eval_by_dpi import size_eval
from .classes.Mole import Mole
from .model_inference.classification_inference import ClassificationModelInference
from .model_inference.segmentation_inference import SegmentationModelInference
from .utils import log, params
from .utils.upload_image import upload_file
from .utils.utils import find_object_coords, find_center_coords, find_object_radius, cut_roi_from_mask
from .utils.params import net_params

app = Flask(__name__)


@app.route("/")
def hello():
    print("Hello World", file=sys.stderr)
    return "Hello"


@app.route("/api/analyze", methods=['POST'])
def analyze():
    image_path = upload_file(request)
    image_path = "/app/files/TestInputs/ISIC_0027334.jpg"
    dpi = request.args['dpi']
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

    moles_analyze_results = {}
    for index, separated_mask in enumerate(segmentation_output):
        lesion_mask = cut_roi_from_mask(separated_mask, find_object_coords(separated_mask))
        border_score, B_score = border_eval(lesion_mask)
        asymmetric_score, A_score = asymmetric_eval(lesion_mask)
        size_score, D_score = size_eval(separated_mask, dpi)
        color_score, C_score = color_eval(resized_image, separated_mask)
        mole_coordinate = find_object_coords(separated_mask)
        mole_center = find_center_coords(mole_coordinate)
        mole_radius = find_object_radius(mole_coordinate)
        final_score = final_evaluation(A_score, B_score, C_score, D_score, classification_score)
        moles_analyze_results[index] = \
            Mole(asymmetric_score, size_score, border_score, color_score, final_score, classification_score,
                 mole_coordinate, mole_center, mole_radius).toJSON()
    return jsonify({'results': json.dumps(moles_analyze_results)})


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=80)
    analyze()
