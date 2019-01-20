import os
import sys
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import keras.layers as layers
from keras.models import Model
from keras import backend as K

import tensorflow as tf
import tensorflow_hub as hub

# enabling the pretrained model for trainig our custom model using tensorflow hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)

DROPOUT = 0.1
# creating a method for embedding and will using method for every input layer
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

# Taking the question1 as input and ceating a embedding for each question before feed it to neural network
q1 = layers.Input(shape=(1,), dtype=tf.string)
embedding_q1 = layers.Lambda(UniversalEmbedding, output_shape=(512,))(q1)
# Taking the question2 and doing the same thing mentioned above, using the lambda function
q2 = layers.Input(shape=(1,), dtype=tf.string)
embedding_q2 = layers.Lambda(UniversalEmbedding, output_shape=(512,))(q2)

# Concatenating the both input layer
merged = layers.concatenate([embedding_q1, embedding_q2])
merged = layers.Dense(200, activation='relu')(merged)
merged = layers.Dropout(DROPOUT)(merged)

# Normalizing the input layer,applying dense and dropout  layer for fully connected model and to avoid overfitting
merged = layers.BatchNormalization()(merged)
merged = layers.Dense(200, activation='relu')(merged)
merged = layers.Dropout(DROPOUT)(merged)

merged = layers.BatchNormalization()(merged)
merged = layers.Dense(200, activation='relu')(merged)
merged = layers.Dropout(DROPOUT)(merged)

merged = layers.BatchNormalization()(merged)
merged = layers.Dense(200, activation='relu')(merged)
merged = layers.Dropout(DROPOUT)(merged)

# Using the Sigmoid as the activation function and binary crossentropy for binary classifcation as 0 or 1
merged = layers.BatchNormalization()(merged)
pred = layers.Dense(2, activation='sigmoid')(merged)
model = Model(inputs=[q1,q2], outputs=pred)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Loading the save weights
model.load_weights('model-04-0.84.hdf5')

def pred(input1, input2):
    print("-----------------------")
    print(input1)
    print("-----------------------")
    print(input2)
    q1 = input1
    q1 = np.array([[q1],[q1]])
    q2 = input2
    q2 = np.array([[q2],[q2]])

    # Using the same tensorflow session for embedding the test string
    with tf.Session() as session:
      K.set_session(session)
      session.run(tf.global_variables_initializer())
      session.run(tf.tables_initializer())
      # Predicting the similarity between the two input questions

      predicts = model.predict([q1, q2], verbose=0)
      predict_logits = predicts.argmax(axis=1)
      print("---------------")
      print(predicts)
      print("---------------")
      if(predict_logits[0] == 1):
        return "Similar"
      else:
        return "Not Similar"
