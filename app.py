from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configuration
class Config:
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    MODEL_PATH = os.getenv('MODEL_PATH', '.')

config = Config()
app.config.from_object(config)

# helper for loading artifacts
def load_pickle(filename: str) -> Any:
    """Load a pickle file with error handling."""
    filepath = os.path.join(config.MODEL_PATH, filename)
    if not os.path.exists(filepath):
        error_msg = f"Required file '{filename}' not found. Run 'python train_model.py' to generate the model and preprocessing objects."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        error_msg = f"Error loading {filename}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

# Load model, scaler, and column names
try:
    model = load_pickle('svm_model.pkl')
    scaler = load_pickle('scaler.pkl')
    model_columns = load_pickle('model_columns.pkl')
    logger.info("Model artifacts loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    model = None
    scaler = None
    model_columns = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'columns_loaded': model_columns is not None
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        if not all([model, scaler, model_columns]):
            return jsonify({'error': 'Model not loaded'}), 500

        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        prediction_text, confidence = predict_claim(data)
        return jsonify({
            'prediction': prediction_text,
            'is_fraud': 'Fraudulent' in prediction_text,
            'confidence': confidence
        })
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not all([model, scaler, model_columns]):
            result = "Error: Model not loaded. Please run train_model.py first."
            confidence = None
        else:
            data = request.form.to_dict()
            result, confidence = predict_claim(data)

    except Exception as e:
        logger.error(f"Web prediction error: {str(e)}")
        result = f"Error: {str(e)}"
        confidence = None

    return render_template('index.html', prediction_text=result, confidence=confidence)

def predict_claim(data: Dict[str, Any]) -> tuple[str, Optional[float]]:
    """Core prediction logic."""
    # 1. FIX COLUMN MISMATCH & GET THE 140 VALID COLUMNS
    # We strip out the empty '_c39' column so the list exactly matches the scaler
    valid_columns = [col for col in model_columns if col != '_c39']

    # 2. Initialize with the scaler's mean (average) values!
    # Instead of 0s, we fill missing data with the exact training averages.
    # This prevents the SVM from penalizing empty fields as extreme outliers.
    input_df = pd.DataFrame([scaler.mean_], columns=valid_columns)

    # 3. Fill Numerical values from the web form
    numerical_fields = [
        'months_as_customer',
        'policy_deductable',
        'total_claim_amount',
        'umbrella_limit',
        'number_of_vehicles_involved'
    ]

    for field in numerical_fields:
        if field in data and field in input_df.columns:
            val = data[field].strip() if isinstance(data[field], str) else str(data[field])
            if val:
                try:
                    input_df.at[0, field] = float(val)
                except ValueError:
                    logger.warning(f"Invalid numerical value for {field}: {val}")

    # 4. Fill Categorical values from the web form
    if 'incident_severity' in data:
        # First, zero out ALL severity columns to erase the "averages"
        for col in input_df.columns:
            if col.startswith("incident_severity_"):
                input_df.at[0, col] = 0

        # Now, set the one selected by the user to 1
        sev_col = "incident_severity_" + str(data['incident_severity'])
        if sev_col in input_df.columns:
            input_df.at[0, sev_col] = 1

    # 5. Scale and Predict
    # (The missing values will perfectly scale to 0.0, which is the exact center of the distribution)
    scaled_data = scaler.transform(input_df.values)
    prediction = model.predict(scaled_data)

    # Get confidence score/probability if available
    confidence = None
    # prefer probability for clearer interpretation
    if hasattr(model, 'predict_proba'):
        # probability of class 1 (fraud)
        prob = model.predict_proba(scaled_data)[0][1]
        confidence = float(prob)
    elif hasattr(model, 'decision_function'):
        # fall back to decision function (not scaled)
        confidence = float(model.decision_function(scaled_data)[0])

    if prediction[0] == 1:
        result = "Fraudulent Claim Detected!"
    else:
        result = "Claim appears to be Genuine."

    return result, confidence

if __name__ == "__main__":
    app.run(
        debug=config.DEBUG,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000))
    )