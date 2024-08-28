import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import getopt

import numpy as np

import keras.preprocessing.image as image

from PIL import Image
from keras.models import load_model as load_keras_model

#from keras.applications import VGG16
#from keras.applications import InceptionV3

#from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import preprocess_input, decode_predictions

import time

'''
import tensorflow as tf
from keras import backend as K

num_cores = 4

num_CPU = 1
num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)
'''


MODEL_DIR = "model"


def predict(model, pil_img):
    try:
        image_size = int(model.input.shape[1])
        img = pil_img.resize((image_size, image_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        prediction_array = model.predict(x)
        pred = decode_predictions(prediction_array, top=3)[0]
        return pred
    except ValueError as err:
        print("Failed to predict ", err)
    return None



def load_model(model_dir):

    for file in os.listdir(model_dir):
        if file.endswith(".model"):
            file = os.path.join(model_dir, file)

            print("Loading model and weights: ", file)
            model = load_keras_model(file)
            print("Model loaded.")

            return model

    return None


def test_prediction_base64(model, imgs_path):
    secondsSinceEpoch = time.time()
    imageLoadTime = 0
    i = 0
    for file in os.listdir(imgs_path):
        file_path = os.path.join(imgs_path, file)
        print("Predict with:", file_path)

        with open(file_path, "rb") as image_file:
            i += 1
            imageLoadStart = time.time()
            pil_img = Image.open(image_file)
            imageLoadTime += time.time() - imageLoadStart

            predict(model, pil_img)
    print("Number of images: ", i)
    print("Sum prediction time: ", (time.time()-secondsSinceEpoch))
    print("Sum image load time: ", imageLoadTime)

def printHelp():
    print(sys.argv[0], '-m <modeldir> -i <imagedir>')

if __name__ == "__main__":
    modeldir = ''
    imagedir = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hm:i:",["modeldir=","imagedir="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit()
        elif opt in ("-m", "--modeldir"):
            modeldir = arg
        elif opt in ("-i", "--imagedir"):
            imagedir = arg
    if modeldir == '' or imagedir == '':
        printHelp();
        sys.exit(1)
    model = load_model(modeldir)

    test_prediction_base64(model, imagedir)
