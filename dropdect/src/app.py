from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import preprocess_input, preprocess_dataset
from model import load_model, retrain_model, evaluate_model

app = Flask(__name__)

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Load the model and scaler once
model_path = 'models/neural_network_model.joblib'
model = joblib.load(model_path)

# Initialize model and scaler
try:
    model = load_model()
    scaler = joblib.load('models/scaler.joblib')
except FileNotFoundError:
    print("Model or scaler not found. Retraining...")
    model, scaler = retrain_model()


@app.route('/', methods=['GET'])
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f4f4f4;
            }
            .container {
                text-align: center;
            }
            h1 {
                font-size: 3em;
                color: #333;
            }
            p {
                font-size: 1.5em;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to the Prediction API!</h1>
            <p>Use the <strong>/predict</strong> endpoint to make predictions.</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    input_data = data['input']
    
    # Reshape input_data to 2D array
    input_data = np.array(input_data).reshape(1, -1)
    
    # Preprocess input data
    preprocessed_data = preprocess_input(input_data, scaler)
    
    # Predict using the model
    prediction = model.predict(preprocessed_data)
    
    return jsonify({'prediction': prediction.tolist()})

@app.route('/evaluate', methods=['POST'])
@app.route('/evaluate', methods=['POST'])
def evaluate_model_endpoint():
    # Load the test data
    try:
        test_data = pd.read_csv('data/test/test.csv')
    except FileNotFoundError:
        return jsonify({"error": "Test data file not found"}), 404

    # Separate features and target
    X_test = test_data.drop('Target', axis=1)
    y_test = test_data['Target']

    # Ensure X_test has the correct number of features
    expected_features = model.n_features_in_
    if X_test.shape[1] != expected_features:
        return jsonify({"error": f"Test data has {X_test.shape[1]} features, but model expects {expected_features} features."}), 400

    # Preprocess the test data
    try:
        preprocessed_data = preprocess_input(X_test.values)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Make predictions
    predictions = model.predict(preprocessed_data)

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
        [1, 8, 5, 2, 1, 1, 1, 13, 10, 6, 10, 1, 0, 0, 1, 1, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10.8, 1.4, 1.74]
    ]
    preprocessed_data = preprocess_input(np.array(test_input))
    prediction = model.predict(preprocessed_data)
    return jsonify({'test_prediction': prediction.tolist()})

# New endpoint: Data Uploading
@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file to the training data directory
    file.save(os.path.join('data/train', file.filename))
    
    return jsonify({"message": "File uploaded successfully"}), 200

# New endpoint: Data Preprocessing
@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    # Call your preprocessing function on the uploaded dataset
    preprocess_dataset(pd.read_csv('data/train/dataset.csv'))
    
    return jsonify({"message": "Data preprocessed successfully"}), 200

# New endpoint: Model Retraining
@app.route('/retrain', methods=['POST'])
def retrain():
    # Retrain the model using the preprocessed dataset
    retrain_model()
    
    return jsonify({"message": "Model retrained successfully"}), 200

# New endpoint: Evaluation After Retraining
@app.route('/evaluate_after_retraining', methods=['POST'])
def evaluate_after_retraining():
    # Evaluate the model after retraining on the test dataset
    metrics = evaluate_model(model, pd.read_csv('data/test/test.csv').drop('Target', axis=1), pd.read_csv('data/test/test.csv')['Target'])
    
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)

