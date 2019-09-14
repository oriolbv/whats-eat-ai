import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import cv2

INPUT_SHAPE_X = 64
INPUT_SHAPE_Y = 64

dirname = os.path.join(os.getcwd(), 'whats-eat-ai\img')
imgpath = dirname + os.sep
 
images = []
directories = []
dircount = []
prevRoot=''
cant=0

print("Reading food images from ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant = cant + 1
            filepath = os.path.join(root, filename)
            image = cv2.imread(filepath)
            (b, g, r)=cv2.split(image)
            image=cv2.merge([r,g,b])
            images.append(image)
            b = "Reading..." + str(cant)
            print (b, end="\r")
            if prevRoot != root:
                print(root, cant)
                prevRoot = root
                directories.append(root)
                dircount.append(cant)
                cant = 0
dircount.append(cant)
 
dircount = dircount[1:]
dircount[0] = dircount[0] + 1
print('Processed directories:', len(directories))
print("Images in directory", dircount)
print('Sum of images:', sum(dircount))

labels=[]
idx=0
for qtt in dircount:
    for i in range(qtt):
        labels.append(idx)
    idx = idx + 1
print("Quantity of labels created: ", len(labels))

food=[]
idx=0
for directory in directories:
    name = directory.split(os.sep)
    print(idx , name[len(name)-1])
    food.append(name[len(name)-1])
    idx=idx+1

# print("images: ", images) bread pudding

y = np.array(labels)
X = np.array(images, dtype=np.uint8)

# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# TRAINING
# Training groups and Testing
train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

INIT_LR = 1e-3
epochs = 6
batch_size = 64

food_model = Sequential()
food_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(INPUT_SHAPE_X,INPUT_SHAPE_Y,3)))
food_model.add(LeakyReLU(alpha=0.1))
food_model.add(MaxPooling2D((2, 2),padding='same'))
food_model.add(Dropout(0.5))

food_model.add(Flatten())
food_model.add(Dense(32, activation='linear'))
food_model.add(LeakyReLU(alpha=0.1))
food_model.add(Dropout(0.5)) 
food_model.add(Dense(nClasses, activation='softmax'))

food_model.summary()

food_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])

sport_train_dropout = food_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

# Save model
food_model.save("food_mnist.h5py")

test_eval = food_model.evaluate(test_X, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])