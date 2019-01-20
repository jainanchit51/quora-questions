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
np.random.seed(10)



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
model.load_weights('model-20-0.85.hdf5')
graph = tf.get_default_graph()

# define the app
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default


# logging for heroku
if 'DYNO' in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.INFO)


# load the model
# model_api = get_model_api()


# API route
@app.route('/api', methods=['POST'])
def api():
    """API function

    All model-specific logic to be defined in the get_model_api()
    function
    """
    input_data = request.json
    app.logger.info("api_input: " + str(input_data))
    output_data = predict(input_data)
    app.logger.info("api_output: " + str(output_data))
    response = jsonify(output_data)
    return response


@app.route('/')
def index():
    index_path = os.path.join(app.static_folder, "index.html")
    return send_file(index_path)

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

def predict(input):
    global graph,model

    q1 = input
    q1 = np.array([[q1],[q1]])
    q2 = input
    q2 = np.array([[q2],[q2]])


    # Using the same tensorflow session for embedding the test string
    with tf.Session() as session:

      K.set_session(session)
      session.run(tf.global_variables_initializer())
      session.run(tf.tables_initializer())
      # Predicting the similarity between the two input questions
      with graph.as_default():
          predicts = model.predict([q1, q2], verbose=0)
          predict_logits = predicts.argmax(axis=1)
          if(predict_logits[0] == 1):
            return "Similar"
          else:
            return "Not Similar"


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=True)
