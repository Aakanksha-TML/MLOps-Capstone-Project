# load packages
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# Load the trained model
with open('lr_model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

# prepare features
def prepare_features(car):
    return pd.DataFrame([{
        'displacement': float(car.get('displacement', 0.0)),
        'cylinders': int(car.get('cylinders', 0)),
        'horsepower': float(car.get('horsepower', 0.0)),
        'weight': float(car.get('weight', 0.0)),
        'acceleration': float(car.get('acceleration', 0.0)),
        'model_year': int(car.get('model_year', 0)),
        'origin': int(car.get('origin', 0))
    }])


# prediction
def predict(features_df):
    preds = model.predict(features_df)
    return float(preds[0])

# api call
app = Flask('auto-mpg-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        car = request.get_json()

        if not car:
            return jsonify({"error": "No input data provided"}), 400

        features_df = prepare_features(car)

        # Handle NaN in case horsepower etc. are missing
        if features_df.isnull().any().any():
            return jsonify({"error": "Missing or invalid values"}), 400

        pred = predict(features_df)

        return jsonify({
            'predicted_mpg': pred
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
