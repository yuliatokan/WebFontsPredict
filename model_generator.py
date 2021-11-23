# -*- coding: utf-8 -*-
"""Lab11-12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TrOtk-XSuCd4ZEehmeOeXPXbmWQ8bju-
"""

import pandas as pd
import tensorflow as tf
import math
import numpy as np
import os
import json
from keras.regularizers import l2

path = "/content/sample_data/fonts"
filenames = os.listdir(path)

data = pd.concat([pd.read_csv(path + '/' + fn) for fn in filenames]).sample(frac=1)
classes = data.font.unique()
classes_dict = {c:i for i,c in enumerate(classes)}
num_of_classes = len(classes)

with open('listfile.txt', 'w') as filehandle:  
    filehandle.writelines("%s\n" % clas for clas in classes)

X = data.iloc[:,12:].values
X = np.true_divide(X,255)

Y = np.array([[1 if y == classes_dict[x] else 0 for y in range(num_of_classes)] for x in data['font'].values])

split = 0.75
splitpoint = int(math.floor(len(X)*split))
X_train, X_test = X[:splitpoint], X[splitpoint:]
Y_train, Y_test = Y[:splitpoint], Y[splitpoint:]
X_train = np.reshape(X_train, (-1,20,20,1))
X_test = np.reshape(X_test, (-1,20,20,1))

model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.0001), input_shape=(20,20,1)),
      tf.keras.layers.MaxPooling2D(2,2),

      tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.0001)),
      tf.keras.layers.MaxPooling2D(2,2),

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(346, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.0001)),
      tf.keras.layers.Dense(num_of_classes, activation='softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, batch_size=64)

model.evaluate(X_test, Y_test, batch_size=64)

model.save("my_model.hdf5")