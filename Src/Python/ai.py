## https://developer.ibm.com/articles/image-recognition-challenge-with-tensorflow-and-keras-pt1/ ##
## train_images and train_lables is training data set. ##
## test_images and test_labels is testing data set for validating the model's performance against unseen data.##

# TensorFlow and tf.keras
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
		app.run(host="localhost", port=5000, debug=True)

@app.route("/") # CHECK THIS line
def hello_world():
    file_name = os.path.join(dirname, 'rrr.png')
    print (file_name)
    return render_template("index.html", title='AI', file_name = file_name)

@app.route("/start", methods = ['GET'])
def start():
#     try:
        # start()
#     except:
#         return redirect('/error')
#     file_name = os.path.join(dirname, 'rrr.png')
#     print (file_name)
#     return render_template("show.html", title='AI', file_name = file_name)
        # return render_template("show.html", title='AI', file_name = "sss")
        return test()

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
        return test()


@app.route("/train", methods = ['GET'])
def routeTrain():
        train()
        return "Training Done"

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      type = request.form['type']
      globing(os.path.join(dirname, 'static/'+type))
      for file in request.files.getlist("file[]"):
        filename = "unknown-"+f"{random.randint(0,100)}"+".png"    
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


# Use Pillow library to convert an input jpeg to a 8 bit grey scale image array for processing.
def greyscaling(path, maxsize):
        img= Image.open(path).convert('L')   # convert image to 8-bit grayscale
        # Make aspect ratio as 1:1, by applying image crop.
    # Please note, croping works for this data set, but in general one
    # needs to locate the subject and then crop or scale accordingly.
        WIDTH, HEIGHT = img.size
        if WIDTH != HEIGHT:
                m_min_d = min(WIDTH, HEIGHT)
                img = img.crop((0, 0, m_min_d, m_min_d))
        # Scale the image to the requested maxsize by Anti-alias sampling.
        img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
        return np.asarray(img)

def load_image_dataset(path_dir, maxsize):
        images = []
        labels = []
        os.chdir(path_dir)
        for file in glob.glob("*.png"):
                img = greyscaling(file, maxsize)
                if re.match('Healthy_Teeth*.*', file):
                        images.append(img)
                        labels.append(0)
                elif re.match('Broken_Teeth*.*', file):
                        images.append(img)
                        labels.append(1)
                elif re.match("unknown*.*",file):
                        images.append(img)
                        labels.append(2)
        return (np.asarray(images), np.asarray(labels))

def display_images(images, labels):
        plt.figure(figsize=(10,10))
        grid_size = min(25, len(images))
        for i in range(grid_size):
                class_names=['Healthy Teeth','Broken Teeth']
                plt.subplot(5, 5, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[labels[i]])
              

maxsize = 100, 100
# Setting up the layers.

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(100, 100)),
        keras.layers.Dense(128, activation=tf.nn.sigmoid),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
])
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
def train():
        (train_images, train_labels) = load_image_dataset(os.path.join(dirname, 'static/train'), maxsize)
        train_images = train_images / 255.0
        model.fit(train_images, train_labels, epochs=500)
        train_loss, train_acc = model.evaluate(train_images, train_labels)
        print('Train accuracy:', train_acc,train_loss)
        return test()

def test():
        (test_images, test_labels) = load_image_dataset(os.path.join(dirname, 'static/test'), maxsize)
        test_images = test_images / 255.0
        model.fit(test_images, test_labels, epochs=100)
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        predictions = model.predict(test_images)
        class_names=['Healthy Teeth','Broken Teeth','Needs training']
        label = np.argmax(predictions, axis = 1)
        #print('Test accuracy:', test_acc,test_loss)
        # for i in range(len(label)):  # BROKEN CODE!
        #         if class_names[label[i-1]] == 'Needs training':
                        
        #         else:
        #                 return class_names[label[i]]


        # return "ACC: " + float(str(test_acc)) + " - LOSS: " + float(str(test_loss))
        # return class_names[label[0]]


        # abd cahges for the return in the end
        # result_array = []
        #   (in for loop)    result_array += [class_names[label[i]],str(float(test_acc))]
        # return result_array


def start():
        (test_images, test_labels) = load_image_dataset(os.path.join(dirname, 'static/test'), maxsize)
        (train_images, train_labels) = load_image_dataset(os.path.join(dirname, 'static/train'), maxsize)
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        model.fit(train_images, train_labels, epochs=500)
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        train_loss, train_acc = model.evaluate(train_images, train_labels)
        predictions = model.predict(test_images)
        predictions_1=model.predict(train_images)
        print('Test accuracy:', test_acc,test_loss)
        print('Train accuracy:', train_acc,train_loss)
       # print(predictions)
       # print(predictions_1)
       # display_images(train_images,np.argmax(predictions_1, axis = 1))
       # plt.show()
        display_images(test_images, np.argmax(predictions, axis = 1))
        saving = plt.savefig(os.path.join(dirname, 'static/rrr.png'))
        img = BytesIO(saving)
        img.seek(0)
        return send_file(img, mimetype='image/*')

start_app() # This should always be the last line of your code.