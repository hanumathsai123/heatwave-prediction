# ============================================================
# Heat Wave Prediction System - Flask Backend
# ============================================================

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# Initialize Flask application
app = Flask(__name__)

# ----------------------------------------------------------
# Load the pre-trained model and scaler from disk
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

model = None
scaler = None
model_loaded = False

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    model_loaded = True
    print("✅ Model and Scaler loaded successfully!")
    
except Exception as e:
    print(f"❌ Warning: Could not load model files: {str(e)}")
    print(f"   Model path: {MODEL_PATH}")
    print(f"   Scaler path: {SCALER_PATH}")

# ----------------------------------------------------------
# Helper: Determine Risk Level
# ----------------------------------------------------------
def get_risk_level(probability, temperature):
    if probability >= 0.80 or temperature >= 44:
        return "HIGH"
    elif probability >= 0.55 or temperature >= 38:
        return "MEDIUM"
    else:
        return "LOW"

# ----------------------------------------------------------
# Helper: Return safety precautions
# ----------------------------------------------------------
def get_precautions(heat_wave, risk_level):
    if not heat_wave:
        return [
            "Weather conditions appear normal. Stay alert for sudden changes.",
            "Keep yourself hydrated throughout the day.",
            "Check local weather forecasts regularly.",
            "Ensure proper ventilation in your home."
        ]

    common = [
        "Drink at least 3–4 litres of water every day.",
        "Avoid going outside between 11 AM and 4 PM.",
        "Wear light-coloured, loose-fitting cotton clothes.",
        "Use sunscreen (SPF 30+) and wear a hat outdoors.",
    ]

    if risk_level == "HIGH":
        return common + [
            "⚠️ Stay indoors as much as possible — critical alert!",
            "Keep elderly, children & pets in cool/AC rooms.",
            "Monitor for symptoms: dizziness, nausea, heavy sweating.",
            "Keep emergency contacts and nearest hospital numbers handy.",
            "Do NOT leave anyone in a parked vehicle.",
        ]
    elif risk_level == "MEDIUM":
        return common + [
            "Take cool showers or baths to reduce body temperature.",
            "Reduce physical activity and rest during peak hours.",
            "Check on neighbours, especially elderly individuals.",
        ]
    else:
        return common + [
            "Stay in shaded areas when outdoors.",
            "Carry a water bottle when going out.",
        ]

# ----------------------------------------------------------
# ROUTE 1: Serve the main HTML page
# ----------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ----------------------------------------------------------
# ROUTE 2: Prediction API Endpoint (POST)
# ----------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded or model is None or scaler is None:
        return jsonify({'error': 'Model is not available. Please check server logs.'}), 503
    
    try:
        data = request.get_json()

        temperature = float(data['temperature'])
        humidity    = float(data['humidity'])
        wind_speed  = float(data['wind_speed'])
        pressure    = float(data['pressure'])

        # Input Validation
        if not (15 <= temperature <= 55):
            return jsonify({'error': 'Temperature must be between 15°C and 55°C'}), 400
        if not (5 <= humidity <= 100):
            return jsonify({'error': 'Humidity must be between 5% and 100%'}), 400
        if not (0 <= wind_speed <= 60):
            return jsonify({'error': 'Wind speed must be between 0 and 60 km/h'}), 400
        if not (990 <= pressure <= 1035):
            return jsonify({'error': 'Pressure must be between 990 and 1035 hPa'}), 400

        # Prepare Input for Model
        features = np.array([[temperature, humidity, wind_speed, pressure]])
        features_scaled = scaler.transform(features)

        # Make Prediction
        prediction  = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Determine Risk Level & Precautions
        heat_wave  = bool(prediction == 1)
        risk_level = get_risk_level(probability, temperature)
        precautions = get_precautions(heat_wave, risk_level)

        # Build Response
        response = {
            'heat_wave':   heat_wave,
            'probability': round(float(probability) * 100, 1),
            'risk_level':  risk_level,
            'precautions': precautions,
            'inputs': {
                'temperature': temperature,
                'humidity':    humidity,
                'wind_speed':  wind_speed,
                'pressure':    pressure,
            }
        }
        return jsonify(response)

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Entry Point
if __name__ == '__main__':
    print("🌡️  Starting Heat Wave Prediction System...")
    print("🔗  Open: http://127.0.0.1:5000")
    if os.environ.get('VERCEL') is None:
        app.run(debug=True, host='0.0.0.0', port=5000)
