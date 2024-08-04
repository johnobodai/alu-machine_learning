from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import preprocess_input, clean_data, scale_data
from prediction import predict

app = Flask(__name__)

# Load the model once
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
    prediction = predict(model, np.array([preprocessed_data]))  # Add extra brackets to match expected input shape
    
    return jsonify({'prediction': prediction.tolist()})

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    # Load the test data
    try:
        test_data = pd.read_csv('data/test/test.csv')  # Update with your actual test data path
    except FileNotFoundError:
        return jsonify({"error": "Test data file not found"}), 404

    # Separate features and target
    X_test = test_data.drop('Target', axis=1)
    y_test = test_data['Target']

    # Preprocess the test data
    preprocessed_data = preprocess_input(X_test.values)

    # Make predictions
    predictions = model.predict(preprocessed_data)

    # Ensure consistent label types
    if isinstance(y_test.iloc[0], str):  # If labels are strings
        y_test = y_test.astype(str)
        predictions = [str(label) for label in predictions]
    else:  # If labels are numeric
        y_test = y_test.astype(float)
        predictions = [float(label) for label in predictions]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Return the results as a JSON response
    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

@app.route('/test_model', methods=['GET'])
def test_model():
    test_input = [
        1, 8, 5, 2, 1, 1, 1, 13, 10, 6, 10, 1, 0, 0, 1, 1, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10.8, 1.4, 1.74
    ]
    preprocessed_data = preprocess_input(test_input)
    prediction = model.predict([preprocessed_data])  # Add extra brackets to match expected input shape
    return jsonify({'test_prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

