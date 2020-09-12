import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import time
import cv2
import numpy as np

UPLOAD_FOLDER = '/files/pictures'
ALLOWED_EXTENSIONS = {'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           getExtenstion(filename) in ALLOWED_EXTENSIONS

def getExtenstion(filename):
    return filename.rsplit('.', 1)[1].lower()

def create_folder(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path) 

def upload_file(request):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'mole_picture' not in request.files:
            flash('No file part')
            return 
        file = request.files['mole_picture']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return
        if file and allowed_file(file.filename):
            filename = secure_filename(str(round(time.time() * 1000)) + "_" + file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                create_folder(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print((os.path.join(app.config['UPLOAD_FOLDER'],filename)), flush=True)
            return filename, os.path.join(app.config['UPLOAD_FOLDER'], filename)

def upload_mask(mask, filename):
    folder_name=os.path.join(app.config['UPLOAD_FOLDER'], "mask")
    if not os.path.exists(folder_name):
        create_folder(folder_name)
    expanded_mask = np.expand_dims(mask, axis = -1)
    cv2.imwrite(os.path.join(folder_name, filename), expanded_mask)
    
