from flask import Flask
from app.algorithms import asymmetric_eval as asymmetric
from app.algorithms import border_eval as border
from app.algorithms import size_eval as size
from app.algorithms import predictions_extractions as prediction
from app.classes import Mole
from flask import jsonify, request

import numpy as np
import numpy.core.multiarray
import cv2
import sys

app = Flask(__name__)


@app.route("/")
def hello():
    print ("Hello World", file=sys.stderr)
    return "Hello"

@app.route("/api/analyze", methods=['POST'])
def analyze():
    print (request.files.get('mask', ''), file=sys.stderr)
    # content = request.get_json()
    # mask = content['mask']
    dpi = request.args['dpi']
    print (dpi, file=sys.stderr)

    # separated_masks = prediction.separate_objects_from_mask(mask)
    # moles = []
    # for index, separated_mask in enumerate(separated_masks):
    #     border = border.border_eval(separated_mask)  
    #     size = size.size_eval(separated_mask, dpi)
    #     asymmetric = asymmetric.asymmetric_eval(separated_mask)
    #     coordinates = border.find_all_coordinates(separated_mask)
    #     moles.append(Mole(asymmetric, size, border, coordinates))
    #     print ("Success", file=sys.stderr)
    return jsonify({'results': dpi})

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=80)
