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
from .utils.upload_image import upload_file, upload_mask
from .utils.utils import find_object_coords, find_center_coords, find_object_radius, cut_roi_from_mask,\
    verify_segmentation_mask, normalize_final_score, align_by_centroid
from .utils.params import net_params
from werkzeug.exceptions import HTTPException
from easydict import EasyDict as edict
app = Flask(__name__)


@app.errorhandler(Exception)
def handle_exception(e):
    response = edict()
    # replace the body with JSON
    response.data = json.dumps({
        "description": str(e),
    })
    response.content_type = "application/json"

    # now you're handling non-HTTP exceptions only
    return response, 500


@app.route("/api/analyze", methods=['POST'])
def analyze():
    filename, image_path = upload_file(request)
    dpi = request.args['dpi']
    log.writeToLogs("Starting to check a new image: " + filename)

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
    upload_mask(segmentation_output, filename)

    # evaluation
    classification_score = classification_eval(classification_output)
    verify_segmentation_mask(segmentation_output)

    moles_analyze_results = {}
    for index, separated_mask in enumerate(segmentation_output):
        aligned_mask = align_by_centroid(separated_mask)
        lesion_mask_aligned = cut_roi_from_mask(aligned_mask, find_object_coords(aligned_mask))
        border_score, B_score = border_eval(lesion_mask_aligned)
        asymmetric_score, A_score = asymmetric_eval(lesion_mask_aligned)
        size_score, D_score = size_eval(separated_mask, dpi)
        color_score, C_score = color_eval(resized_image, separated_mask)
        mole_coordinate = find_object_coords(separated_mask)
        mole_center = find_center_coords(separated_mask)
        mole_radius = find_object_radius(mole_center, mole_coordinate)
        final_score = final_evaluation(A_score, B_score, C_score, D_score, classification_score)
        moles_analyze_results[index] = \
            Mole(asymmetric_score, size_score, border_score, color_score, normalize_final_score(final_score),
                 classification_score, mole_center, mole_radius).toJSON()
    return json.dumps(moles_analyze_results)


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=80)
    analyze()
