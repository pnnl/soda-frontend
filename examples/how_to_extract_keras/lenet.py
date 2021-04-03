# import the necessary packages

import numpy as np
import tensorflow.keras.utils as np_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from collections import OrderedDict

# load minit data
from tensorflow.keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# plot 6 images as gray scale
# import matplotlib.pyplot as plt

# reshape the data to four dimensions, due to the input of model
# reshape to be [samples][width][height][pixels]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# parameters
EPOCHS = 1
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 10
norm_size = 28

# start to train model

print('start to train model')
# define lenet model


def l_model(width, height, depth, NB_CLASS):

    model = Sequential()

    inputShape = (height, width, depth)

    # if we are using "channels last", update the input shape

    if K.image_data_format() == "channels_first":  # for tensorflow

        inputShape = (depth, height, width)

    # first set of CONV => RELU => POOL layers

    model.add(InputLayer(input_shape=inputShape))

    model.add(Conv2D(6, (5, 5), input_shape=inputShape,  use_bias=False, padding="same"))

    model.add(Activation("relu"))

    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers

    model.add(Conv2D(16, (5, 5),  use_bias=False))

    model.add(Activation("relu"))

    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers

    model.add(Flatten())

    model.add(Dense(500,  use_bias=False))

    model.add(Activation("relu"))



    # softmax classifier

    model.add(Dense(NB_CLASS,  use_bias=False))

    model.add(Activation("softmax"))



    # return the constructed network architecture

    return model



model = l_model(width=norm_size, height=norm_size, depth=1, NB_CLASS=CLASS_NUM)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Use generators to save memory
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),steps_per_epoch=len(x_train) // BS, epochs=EPOCHS, verbose=2)

te_loss, te_accuracy = model.evaluate(x_test, y_test)

# save model by json and weights
# save json
from tensorflow.keras.models import model_from_json
json_string = model.to_json()
with open(r'lenet.json', 'w') as file:
    file.write(json_string)

# save weights
model.save_weights('lenet.h5')
model.summary()
