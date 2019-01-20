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

import quora_model

import tensorflow as tf
import tensorflow_hub as hub



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
    input1 = request.json["input1"]
    input2 = request.json["input2"]
    output_data = predict(input1, input2)
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

def predict(input1,input2):
    return quora_model.pred(input1,input2)


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=True)
