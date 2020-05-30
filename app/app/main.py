from flask import Flask
from .algorithms import asymmetric_eval as asy
from .algorithms import border_eval as border
from .algorithms import size_eval as size
from .algorithms import predictions_extractions as prediction
from .classes.Mole import Mole
from .utils import upload_image as ui
from flask import jsonify, request

import numpy as np
import numpy.core.multiarray
import cv2
import sys
import jsonpickle

app = Flask(__name__)

@app.route("/")
def hello():
    print ("Hello World", file=sys.stderr)
    return "Hello"

@app.route("/api/analyze", methods=['POST'])
def analyze():
    path = ui.upload_file(request)
    dpi = request.args['dpi']
    # file = request.files['mask']
    print (path, file=sys.stderr)
    mask = cv2.imread(path,  -1)
    separated_masks = prediction.separate_objects_from_mask(mask)
    moles = []
    for index, separated_mask in enumerate(separated_masks):
        # smt = "/files/seperated_masks/"+ file.filename
        # cv2.imwrite(smt, separated_mask)
        bdr = border.border_eval(separated_mask)  
        sz = size.size_eval(separated_mask, int(dpi))
        asymtrc = asy.eval_asymmetric(separated_mask)
        crdint = border.find_all_coordinates(separated_mask)
        moles.append(Mole(asymtrc, sz, bdr, crdint))
    print (moles[0].toJSON(), file=sys.stderr)
    return jsonify({'results': moles[1000].toJSON()})

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=80)
