# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:07:35 2020

@author: eponr
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer



# Define a flask app
app = Flask(__name__)

# Define a flask app
app = Flask(__name__)
UPLOAD_FOLDER   = "C:\\Users\\eponr\\Desktop\\Open_Source_project\\Flower_Classfication_app\\static"
MODEL_PATH      = "C:\\Users\\eponr\\Desktop\\Open_Source_project\\Flower_Classfication_app\\model\\Flower_classification_model_vgg16_88percent.hdf5"
img_path        = 'C:\\Users\\eponr\\Desktop\\Open_Source_project\\Flower_Classfication_app\\uploads\\image_0887.jpg'



def predict(img_path= None, model_path = None):
    class_names = {0: 'Bluebell', 1: 'Buttercup', 2: "Colts'Foot", 3: 'Cowslip', 
                   4: 'Crocus', 5: 'Daffodil', 6: 'Daisy', 7: 'Dandelion', 8: 'Fritillary', 
                   9: 'Iris', 10: 'LilyValley', 11: 'Pansy', 12: 'Snowdrop', 13: 'Sunflower', 
                   14: 'Tigerlily', 15: 'Tulip', 16: 'Windflower'}

    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x * 1./255
    x = x[np.newaxis, ...]
#    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)#, mode='caffe')
    
    MODEL = load_model(model_path)
    
    preds = np.argmax(MODEL.predict(images))
    preds = class_names[preds]
    
    return preds 



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = predict(file_path, MODEL_PATH)
        

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return pred
    return None


if __name__ == '__main__':
    app.run(debug=True)

