import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from skimage.transform import resize
from flask import Flask

app = Flask(__name__)

@app.route("/")
def classify():
    classes = [ 'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad',
                'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito']
    model = load_model('food_mnist.h5py')
    images=[]
    filenames = ['img_carpaccio.jpg','img_apple_pie.jpg']
    for filepath in filenames:
        image = plt.imread(filepath)
        image_resized = resize(image, (64, 64),preserve_range=True)
        images.append(image_resized)

    # List to numpy array
    X = np.array(images, dtype=np.uint8)
    test_X = X.astype('float32')
    test_X = test_X / 255.
    predicted_classes = model.predict(test_X)
    prediction_1 = predicted_classes[0]
    prediction_2 = predicted_classes[1]
    print("Prediction 1: ", prediction_1)
    print("Prediction 2: ", prediction_2)
    actual_idx = 0
    best_coincidence_idx = 0
    greater_value = 0
    for pred in prediction_1:
        if greater_value < pred:
            greater_value = pred
            best_coincidence_idx = actual_idx
        actual_idx += 1

    return  "Your plate is " + classes[best_coincidence_idx] + "!!! The prediction value is " + str(greater_value)

if __name__ == "__main__":
    app.run(debug=True)
