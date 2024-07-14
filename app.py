from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import logging

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained models
with open('./model/decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('./model/random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO)

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
        
        # Handle the features based on the selected model
        if model_name == 'Decision Tree':
            features = ['ratio_to_median_purchase_price', 'online_order', 'distance_from_last_transaction', 'distance_from_home', 'repeat_retailer']
        elif model_name == 'Random Forest':
            features = ['ratio_to_median_purchase_price', 'online_order', 'distance_from_last_transaction', 'distance_from_home', 'repeat_retailer']
        else:
            return jsonify({'error': 'Invalid model name'}), 400
        
        # Extract the required features
        X = pd.DataFrame([data], columns=features)
        
        # Handle missing values
        X = X.fillna(-999).infer_objects(copy=False)
        
        # Make prediction based on the selected model
        if model_name == 'Decision Tree':
            prediction = decision_tree_model.predict(X)
        elif model_name == 'Random Forest':
            prediction = random_forest_model.predict(X)
        
        # Convert prediction to descriptive result
        result = 'Penipuan' if prediction[0] == 1 else 'Bukan Penipuan'
        
        # Log the prediction request and result
        logging.info(f"Prediction request - Data: {data}, Model: {model_name}, Result: {result}")
        
        # Return the result as a JSON response
        return jsonify({'result': result})

    except Exception as e:
        # Log any exception that occurs
        logging.error(f"Prediction failed - Error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)

