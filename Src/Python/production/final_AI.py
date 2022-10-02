#https://developer.ibm.com/articles/image-recognition-challenge-with-tensorflow-and-keras-pt1/
#https://developer.ibm.com/articles/image-recognition-challenge-with-tensorflow-and-keras-pt2/

# NOTE TO developer: Do not change any values or the AI wont work


# TensorFlow and tf.keras
from email.mime import base
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re

# Pillow
import PIL
from PIL import Image
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re

# Pillow
# import PIL
# from PIL import Image
from PIL import *

direname= os.path.dirname(__file__)
maxsize = 50,50
maxsize_w, maxsize_h = maxsize


# Use Pillow library to convert an input jpeg to a 8 bit grey scale image array for processing.
def jpeg_to_8_bit_greyscale(path, maxsize):
        img = Image.open(path).convert('L')   # convert image to 8-bit grayscale
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
# invert_image if true, also stores an invert color version of each image in the training set.
def load_image_dataset(path_dir, maxsize, reshape_size, invert_image=False):
        images = []
        labels = []
        os.chdir(path_dir)
        for file in glob.glob("*.jpg"):
                img = jpeg_to_8_bit_greyscale(file, maxsize)
                inv_image = 255 - img # Generate a invert color image of the original.
                if re.match('Healthy.*', file):
                        images.append(img.reshape(reshape_size))
                        labels.append(0)
                        if invert_image:
                                labels.append(0)
                                images.append(inv_image.reshape(reshape_size))
                elif re.match('Broken.*', file):
                        images.append(img.reshape(reshape_size))
                        labels.append(1)
                        if invert_image:
                                images.append(inv_image.reshape(reshape_size))
        return (np.asarray(images), np.asarray(labels))


def display_images(images, labels,title="Default"):
        plt.title(title)
        plt.figure(figsize=(10,10))
        grid_size = min(25, len(images))
        for i in range(grid_size):
                plt.subplot(5, 5, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[labels[i]])




class_names = ['Healthy','Broken']

# Setting up the layers.
bigger_model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
                keras.layers.Dense(512, activation=tf.nn.sigmoid),
                keras.layers.Dense(128, activation=tf.nn.sigmoid),
                keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(2, activation=tf.nn.softmax)
        ])

bigger_model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy','sparse_categorical_crossentropy'])

smaller_model = keras.models.Sequential([
                keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
                keras.layers.Dense(64, activation=tf.nn.sigmoid),
                keras.layers.Dense(2, activation=tf.nn.softmax)
                ])
smaller_model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy','sparse_categorical_crossentropy'])

def train():

        (train_images, train_labels) = load_image_dataset(
	path_dir= os.path.join(direname, 'static/train'),
	maxsize=maxsize,
	reshape_size=(maxsize_w, maxsize_h, 1),
	invert_image=False)

        (test_images, test_labels) = load_image_dataset(
	path_dir= os.path.join(direname, 'static/train_test'),
	maxsize=maxsize,
	reshape_size=(maxsize_w, maxsize_h, 1),
	invert_image=False)

        train_images = train_images / 255.0
        test_images = test_images / 255.0


        # KERAS preprocessing of image and ImageDataGenerator
        datagen = keras.preprocessing.image.ImageDataGenerator(
                zoom_range=0.2, # randomly zoom into images
                featurewise_center=True,
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True)  # randomly flip images
        datagen.fit(train_images)

        bigger_model.fit_generator(datagen.flow(train_images, train_labels),
                epochs=30,
                validation_data=(train_images, train_labels),
                verbose=2,
                workers=4)

        smaller_model.fit_generator(datagen.flow(train_images, train_labels),
                epochs=30,
                validation_data=(train_images, train_labels),
                verbose=2,
                workers=4)
        # smaller_model.fit(train_images, train_labels,
        # 	epochs=100,
        # 	validation_data=(test_images, test_labels),
        # 	verbose=2,
        # 	workers=4)

def test():

        (test_images, test_labels) = load_image_dataset(
	path_dir= os.path.join(direname, 'static/test'),
	maxsize=maxsize,
	reshape_size=(maxsize_w, maxsize_h, 1),
	invert_image=False)

        test_images = test_images / 255.0

        test_acc=bigger_model.evaluate(test_images, test_labels)
        test_acc_small=smaller_model.evaluate(test_images, test_labels)
        predictions= bigger_model.predict(test_images)
        small_prediction=smaller_model.predict(test_images)
        print(bigger_model.summary())
        print('Test accuracy:', test_acc)
        print('Test for small model accuracy:', test_acc_small)
        print(predictions)
        print(smaller_model.summary())

        textLabels = np.argmax(predictions, axis = 1)

        # for x in label:
        #         print("s")
        #         print(x)

        sss = class_names[textLabels[0]]
        result_array = [class_names[textLabels[0]], str(float(test_acc[0]))]
        # in case of multible test images use this code instead
        # result_array = []
        #   (in for loop)    result_array += [class_names[label[i]],str(float(test_acc))]

        print(class_names[textLabels[0]])
        print(str(float(test_acc[0])))
        return result_array

        #display_images(test_images, test_labels, title="Test Images")
        # display_images(test_images.reshape((len(test_images), maxsize_w, maxsize_h)),np.argmax(predictions, axis = 1), title = "small")
        # plt.show()

# train()
# test()


