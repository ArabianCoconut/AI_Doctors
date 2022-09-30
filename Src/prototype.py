## https://developer.ibm.com/articles/image-recognition-challenge-with-tensorflow-and-keras-pt1/ ##
## train_images and train_lables is training data set. ##
## test_images and test_labels is testing data set for validating the model's performance against unseen data.##

# TensorFlow and tf.keras
from email.mime import base
from matplotlib import *
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np  
import matplotlib.pyplot as plt
import glob, os
import re
import json
import random
# Pillow
import PIL
from PIL import Image
import os
import glob
from io import BytesIO

import base64
import json

# print os.getcwd()
dirname = os.path.dirname(__file__)
print (os.getcwd())
print (dirname)
print (os.path.join(os.getcwd(), 'test'))
print (__file__)

from flask import *
app = Flask(__name__)

def start_app():
		app.run(host="localhost" , port=5000, debug=True)

@app.route("/") # CHECK THIS line
def hello_world():
    file_name = os.path.join(dirname, 'rrr.png')
    print (file_name)
    return render_template("index.html", title='AI', file_name = file_name)

@app.route("/start", methods = ['GET'])
def start():
    return redirect(url_for('RoundTest'))
#     try:
        # start()
#     except:
#         return redirect('/error')
#     file_name = os.path.join(dirname, 'rrr.png')
#     print (file_name)
#     return render_template("show.html", title='AI', file_name = file_name)
        # return render_template("show.html", title='AI', file_name = "sss")
    # return render_template("show.html", title='AI', file_name = "sss")
    r

@app.route("/test", methods = ['GET'])
def RoundTest():
#     try:
        # start()
#     except:
#         return redirect('/error')
#     file_name = os.path.join(dirname, 'rrr.png')
#     print (file_name)
#     return render_template("show.html", title='AI', file_name = file_name)
        # return render_template("show.html", title='AI', file_name = "sss")
       return everything()


@app.route("/train", methods = ['GET'])
def routeTrain():
        #train()
        return "Training Done"

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      type = request.form['type']
      globing(os.path.join(dirname, 'static/'+type))
      for file in request.files.getlist("file[]"):
        filename = "unknown-"+f"{random.randint(0,100)}"+".jpg"  
        file.save(os.path.join(dirname, 'static/'+type+'/'+filename))
      return redirect('/')

@app.route('/api/uploader', methods = ['GET', 'POST'])
def api_upload_file():

   if request.method == 'POST':

      img_base64 = request.form['img']
      img_base64 = bytes(img_base64, encoding='utf-8')
      globing(os.path.join(dirname, 'static/test'))
      with open("static/test/imageToSave.png", "wb") as fh:
        fh.write(base64.decodebytes(img_base64))

      result = test()
      data_rx={"Result":f"{result[0][0]}","Accuracy":f"{result[0][0]}"}

      return json.dumps(data_rx)

@app.route("/error")
def error():
    file_name = os.path.join(dirname, 'rrr.png')
    print (file_name)
    return render_template("error.html", title='AI', file_name = file_name)



def globing(path):
    files = glob.glob(path+'/*')
    for f in files:
        os.remove(f)


def jpeg_to_8_bit_greyscale(path, maxsize):
	img = Image.open(path).convert('L')   # convert image to 8-bit grayscale

	# Make aspect ratio as 1:1, by applying image crop.
    # Please note, croping works for this dataset, but in general one 
    # needs to locate the subject and then crop or scale accordingly.
	WIDTH, HEIGHT = img.size
	if WIDTH != HEIGHT:
		m_min_d = min(WIDTH, HEIGHT)
		img = img.crop((0, 0, m_min_d, m_min_d))
	# Scale the image to the requested maxsize by Anti-alias sampling.
	img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
	img_rotate = img.rotate(90)
	print("rotating...")
	print(img_rotate.size)
	return (np.asarray(img), np.asarray(img_rotate))

class_names = ['Healthy', 'Broken',"unknown"]
##
# invert_image if true, also stores an invert color version of each image in the training set.
def load_image_dataset(path_dir, maxsize, reshape_size, invert_image=False):
	images = []
	labels = []
	os.chdir(path_dir)
	for file in glob.glob("*.jpg"):
		(img, img_rotate) = jpeg_to_8_bit_greyscale(file, maxsize)
		inv_image = 255 - img
		if re.match('Healthy.*', file):
			images.append(img.reshape(reshape_size))
			labels.append(0)
			if invert_image:
				images.append(inv_image.reshape(reshape_size))
				images.append(img_rotate.reshape(reshape_size))
				labels.append(0)
				labels.append(0)
		elif re.match('Broken.*', file):
			images.append(img.reshape(reshape_size))
			labels.append(1)
			if invert_image:
				images.append(inv_image.reshape(reshape_size))
				images.append(img_rotate.reshape(reshape_size))
				labels.append(1)
				labels.append(1)
		elif re.match('unknown.*', file):
			images.append(img.reshape(reshape_size))
			labels.append(2)
			if invert_image:
				images.append(inv_image.reshape(reshape_size))
				images.append(img_rotate.reshape(reshape_size))
				labels.append(2)
				labels.append(2)
	return (np.asarray(images), np.asarray(labels))

def load_test_set(path_dir, maxsize):
	test_images = []
	os.chdir(path_dir)
	for file in glob.glob("*.jpg"):
		img = jpeg_to_8_bit_greyscale(file, maxsize)
		test_images.append(img)
	return (np.asarray(test_images))


def everything():
    maxsize = 50, 50
    maxsize_w, maxsize_h = maxsize

    (train_images, train_labels) = load_image_dataset(
        path_dir= os.path.join(dirname, 'static/train'),
        maxsize=maxsize,
        reshape_size=(maxsize_w, maxsize_h, 1),
        invert_image=False)

    (test_images, test_labels) = load_image_dataset(
        path_dir= os.path.join(dirname, 'static/test'),
        maxsize=maxsize,
        reshape_size=(maxsize_w, maxsize_h, 1),
        invert_image=False)



    # scaling the images to values between 0 and 1.
    train_images = train_images / 255.0

    test_images = test_images / 255.0


    # Setting up the layers.

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.04, nesterov=True)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
        keras.layers.Dense(128, activation=tf.nn.sigmoid),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])


    model.compile(optimizer=sgd, 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=1000)

    test_loss, test_acc = model.evaluate(test_images, test_labels)


    predictions = model.predict(test_images)
    data = {"test_acc":test_acc,"test_loss":test_loss}
    return data


start_app() # This should always be the last line of your code.