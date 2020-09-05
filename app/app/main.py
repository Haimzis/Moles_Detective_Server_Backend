from flask import Flask, jsonify, request
import sys
from .algorithms.asymmetric_eval import asymmetric_eval
from .algorithms.border_eval import border_eval 
from .algorithms.size_eval import size_eval
from .algorithms.color_eval import color_eval
from .algorithms.classification_eval import classification_eval
from .algorithms.final_evaluation import final_evaluation
from .classes.Mole import Mole
from .model_inference import segmentation_inference
from .model_inference import classification_inference
from .utils import log, params
from .utils.upload_image import upload_file
from .utils.utils import find_object_coords, find_center_coords, find_object_radius
from .model_inference.classification_inference import ClassificationModelInference
from .model_inference.segmentation_inference import SegmentationModelInference

app = Flask(__name__)

@app.route("/")
def hello():
    print ("Hello World", file=sys.stderr)
    return "Hello"

@app.route("/api/analyze", methods=['POST'])
def analyze():
    image_path = upload_file(request)
    dpi = request.args['dpi']
    log.writeToLogs("Starting to check a new image: "+image_path)
    # separated_masks = prediction.separate_objects_from_mask(mask) TODO: in the future we will separate more than one mask
    classification_inference_instance = ClassificationModelInference(params.input_tensor_name, params.output_tensor_name, params.INPUT_SIZE, params.frozen_model_name,
                 params.frozen_model, None, params.batch_size)
    resized_image, classification_output = classification_inference_instance.quick_inference(image_path)
    segmentation_inference_instance = SegmentationModelInference(params.input_tensor_name, params.output_tensor_name, params.INPUT_SIZE, params.frozen_model_name,
                 params.frozen_model, None)
    resized_image, segmentation_output = segmentation_inference_instance.quick_inference(image_path)
    classification_score = classification_eval(classification_output)

    moles_analyze_results = []
    for index, separated_mask in enumerate(segmentation_output):
        border_score = border_eval(separated_mask)  
        size_score = size_eval(separated_mask, int(dpi))
        asymmetric_score = asymmetric_eval(separated_mask)  
        color_score = color_eval(resized_image)
        mole_coordinate = find_object_coords(separated_mask)
        mole_center = find_center_coords(mole_coordinate)
        mole_radius = find_object_radius(mole_coordinate)
        final_score = final_evaluation(border_score, size_score, asymmetric_score, color_score, classification_score)
        moles_analyze_results.append(Mole(asymmetric_score, size_score, border_score, color_score, final_score, classification_score, mole_coordinate, mole_center, mole_radius))
    return jsonify({'results': moles_analyze_results.toJSON()})

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=80)
