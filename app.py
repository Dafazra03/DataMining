import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import logging

# Initialize the Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Function to get the correct path for models
def get_model_path(file_name):
    if os.path.exists('/home/another-scheme-nuo/model/'):
        return os.path.join('/home/another-scheme-nuo/model/', file_name)
    else:
        return os.path.join('model', file_name)

# Load pre-trained models
try:
    with open(get_model_path('decision_tree_model.pkl'), 'rb') as file:
        decision_tree_model = pickle.load(file)
    logging.info("Decision Tree model loaded successfully")
except Exception as e:
    decision_tree_model = None
    logging.error(f"Failed to load Decision Tree model: {str(e)}")

try:
    with open(get_model_path('random_forest_model.pkl'), 'rb') as file:
        random_forest_model = pickle.load(file)
    logging.info("Random Forest model loaded successfully")
except Exception as e:
    random_forest_model = None
    logging.error(f"Failed to load Random Forest model: {str(e)}")

# Home route to render HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json

        # Get selected model from the request
        model_name = data.get('model_name')

        # Define the features based on the selected model
        features = ['ratio_to_median_purchase_price', 'online_order', 'distance_from_last_transaction', 'distance_from_home', 'repeat_retailer']

        # Extract the required features
        X = pd.DataFrame([data], columns=features)

        # Handle missing values
        X = X.fillna(-999).infer_objects(copy=False)

        # Make prediction based on the selected model
        if model_name == 'Decision Tree' and decision_tree_model:
            prediction = decision_tree_model.predict(X)
            accuracy = decision_tree_model.score(X, [prediction[0]]) * 100  # Calculate accuracy and convert to percentage
        elif model_name == 'Random Forest' and random_forest_model:
            prediction = random_forest_model.predict(X)
            accuracy = random_forest_model.score(X, [prediction[0]]) * 100  # Calculate accuracy and convert to percentage
        else:
            return jsonify({'error': 'Invalid model name or model not loaded'}), 400

        # Convert prediction to descriptive result
        result = 'Penipuan' if prediction[0] == 1 else 'Bukan Penipuan'

        # Log the prediction request and result
        logging.info(f"Prediction request - Data: {data}, Model: {model_name}, Result: {result}, Accuracy: {accuracy}%")

        # Return the result as a JSON response
        return jsonify({'result': result, 'accuracy': f"{accuracy:.2f}%"})

    except Exception as e:
        # Log any exception that occurs
        logging.error(f"Prediction failed - Error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
