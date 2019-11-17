import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    # classes = [ 'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad',
    #             'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito']


    classes = [ "apple_pie",
                "baby_back_ribs",
                "baklava",
                "beef_carpaccio",
                "beef_tartare",
                "beet_salad",
                "beignets",
                "bibimbap",
                "bread_pudding",
                "breakfast_burrito",
                "bruschetta",
                "caesar_salad",
                "cannoli",
                "caprese_salad",
                "carrot_cake",
                "ceviche",
                "cheesecake",
                "cheese_plate",
                "chicken_curry",
                "chicken_quesadilla",
                "chicken_wings",
                "chocolate_cake",
                "chocolate_mousse",
                "churros",
                "clam_chowder",
                "club_sandwich",
                "crab_cakes",
                "creme_brulee",
                "croque_madame",
                "cup_cakes",
                "deviled_eggs",
                "donuts",
                "dumplings",
                "edamame",
                "eggs_benedict",
                "escargots",
                "falafel",
                "filet_mignon",
                "fish_and_chips",
                "foie_gras",
                "french_fries",
                "french_onion_soup",
                "french_toast",
                "fried_calamari",
                "fried_rice",
                "frozen_yogurt",
                "garlic_bread",
                "gnocchi",
                "greek_salad",
                "grilled_cheese_sandwich",
                "grilled_salmon",
                "guacamole",
                "gyoza",
                "hamburger",
                "hot_and_sour_soup",
                "hot_dog",
                "huevos_rancheros",
                "hummus",
                "ice_cream",
                "lasagna",
                "lobster_bisque",
                "lobster_roll_sandwich",
                "macaroni_and_cheese",
                "macarons",
                "miso_soup",
                "mussels",
                "nachos",
                "omelette",
                "onion_rings",
                "oysters",
                "pad_thai",
                "paella",
                "pancakes",
                "panna_cotta",
                "peking_duck",
                "pho",
                "pizza",
                "pork_chop",
                "poutine",
                "prime_rib",
                "pulled_pork_sandwich",
                "ramen",
                "ravioli",
                "red_velvet_cake",
                "risotto",
                "samosa",
                "sashimi",
                "scallops",
                "seaweed_salad",
                "shrimp_and_grits",
                "spaghetti_bolognese",
                "spaghetti_carbonara",
                "spring_rolls",
                "steak",
                "strawberry_shortcake",
                "sushi",
                "tacos",
                "takoyaki",
                "tiramisu",
                "tuna_tartare",
                "waffles"
]
    model = load_model('food_mnist_64_64.h5py')
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
    # print(len(classes), best_coincidence_idx)
    # print(classes)
    # print(prediction)

    return  "Your plate is " + classes[best_coincidence_idx] + "!!! The prediction value is " + str(greater_value)

if __name__ == "__main__":
    classify()
    #app.run(debug=True)
