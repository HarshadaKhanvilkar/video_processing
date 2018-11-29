# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:13:36 2018

@author: Harshada
"""
import numpy as np
import os
from flask import Flask, request, redirect, url_for,jsonify
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras import backend as K
import tensorflow as tf
from flask_restplus import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='Sample API',
    description='A sample API',
)

UPLOAD_FOLDER = 'D:\\CBA\\capstone\\scrap'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# dimensions of our images
img_width, img_height = 299, 299

global graph
graph = tf.get_default_graph() 
def get_prediction(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    with graph.as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            model = load_model('D:\\CBA\\capstone\\scrap\\inception-model-v2.h5')
            preds = model.predict(x)
    print(preds[0])
    return preds[0].tolist()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api.route('/predict')
class MyResource(Resource):
    def get(self):
        K.clear_session()
        img = image.load_img('D:\\CBA\\capstone\\scrap\\pose2.jpg', target_size=(img_width, img_height))
        pred = get_prediction(img)    
        return jsonify({'prediction': pred})

@api.route('/files')
class MyResource(Resource):
    def get(self):
        K.clear_session()
        img = image.load_img('D:\\CBA\\capstone\\scrap\\pose2.jpg', target_size=(img_width, img_height))
        pred = get_prediction(img)    
        return jsonify({'prediction': pred})
    
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return 'uploaded_file sucessfully '
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)