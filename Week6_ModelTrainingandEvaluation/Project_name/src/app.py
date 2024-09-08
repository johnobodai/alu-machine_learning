from flask import Flask, request, jsonify
import joblib
import numpy as np
from preprocessing import preprocess_input
from prediction import predict

app = Flask(__name__)

# Load the neural network model
model_path = 'models/neural_network_model.joblib'
model = joblib.load(model_path)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the prediction API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    input_data = data['input']

    # Preprocess input data
    preprocessed_data = preprocess_input(np.array(input_data))

    # Predict using the model
    prediction = predict(model, preprocessed_data)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

