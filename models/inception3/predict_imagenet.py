import os

import numpy as np

import keras.preprocessing.image as image

from PIL import Image
from keras.models import load_model as load_keras_model

from keras.applications.inception_v3 import preprocess_input, decode_predictions

MODEL_DIR = "model"


def predict(model, pil_img):
    image_size = int(model.input.shape[1])

    img = pil_img.resize((image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prediction_array = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    pred = decode_predictions(prediction_array, top=3)[0]

    print(str(pred))
    print('Predicted:', pred[0][1])

    return pred


def load_model():
    model_dir = MODEL_DIR

    for file in os.listdir(model_dir):
        if file.endswith(".model"):
            file = os.path.join(model_dir, file)

            print("Loading model and weights: " + file)
            model = load_keras_model(file)
            print("Model loaded.")

            return model

    return None

