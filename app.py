import os
import traceback
from functools import wraps

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# ==========================================================
# LOAD MODEL
# ==========================================================

from pathlib import Path
from simulation_api import simulation
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024
app.register_blueprint(simulation)
BASE = Path(__file__).resolve().parent
model = None
feature_columns = ["mean", "std", "variance", "max", "min", "range", "energy"]
# Legacy model is loaded only for legacy requests, so its version cannot block simulation.
API_KEY = os.environ.get("API_KEY")

print("===================================================")
print(" Pulse Diagnosis Backend Started")
print("===================================================")
print("Loaded Features :", feature_columns)
print("API KEY Loaded :", API_KEY is not None)
print("===================================================")


# ==========================================================
# API KEY AUTHENTICATION
# ==========================================================

def require_api_key(f):

    @wraps(f)
    def decorated(*args, **kwargs):

        key = request.headers.get("x-api-key")

        if API_KEY is None:

            return jsonify({
                "success": False,
                "message": "Server API key is missing."
            }), 500

        if key != API_KEY:

            return jsonify({
                "success": False,
                "message": "Unauthorized"
            }), 401

        return f(*args, **kwargs)

    return decorated


# ==========================================================
# HEALTH
# ==========================================================

@app.route("/api/v1/health", methods=["GET"])
def health():

    return jsonify({

        "success": True,
        "status": "Healthy",

        "model": "Legacy model loaded on demand",
        "simulation": "Available at /api/v1/simulation/predict",
        "version": "2.1-simulation"

    })


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return jsonify({

        "message": "Pulse Diagnosis Backend",

        "status": "Running"

    })


# ==========================================================
# PREDICT
# ==========================================================

@app.route("/api/v1/predict", methods=["POST"])
@require_api_key
def predict():

    try:

        global model, feature_columns
        if model is None:
            try:
                model = joblib.load(BASE / "pulse_model.pkl")
                feature_columns = joblib.load(BASE / "feature_columns.pkl")
            except Exception:
                return jsonify(success=False, message="Legacy model unavailable; simulation uses /api/v1/simulation/predict."), 503
        data = request.get_json()

        if data is None:

            return jsonify({

                "success": False,

                "message": "JSON body not found."

            }), 400

        # ==================================================
        # CHECK REQUIRED FEATURES
        # ==================================================

        for feature in feature_columns:

            if feature not in data:

                return jsonify({

                    "success": False,

                    "message": f"{feature} is missing."

                }), 400

        # ==================================================
        # BUILD FEATURE VECTOR
        # ==================================================

        features = []

        for feature in feature_columns:

            features.append(float(data[feature]))

        features = np.array(features).reshape(1, -1)

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = int(model.predict(features)[0])

        probability = model.predict_proba(features)[0]

        confidence = float(np.max(probability) * 100)

        # ==================================================
        # LABEL MAPPING
        # ==================================================

        if prediction == 0:

            diagnosis = "Normal"

            risk = "Low"

            recommendation = (
                "Pulse pattern appears normal. "
                "Continue regular monitoring."
            )

        else:

            diagnosis = "Abnormal"

            risk = "High"

            recommendation = (
                "Abnormal pulse pattern detected. "
                "Consider consulting a healthcare professional."
            )

        # ==================================================
        # RESPONSE
        # ==================================================

        return jsonify({

            "success": True,

            "prediction_id": prediction,

            "prediction": diagnosis,

            "confidence": round(confidence, 2),

            "risk_level": risk,

            "recommendation": recommendation,

            "model": "Extra Trees",

            "features": {

                "mean": float(data["mean"]),

                "std": float(data["std"]),

                "variance": float(data["variance"]),

                "max": float(data["max"]),

                "min": float(data["min"]),

                "range": float(data["range"]),

                "energy": float(data["energy"])

            }

        }), 200

    except ValueError:

        return jsonify({

            "success": False,

            "message": "All features must be numeric."

        }), 400

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500


# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(os.environ.get("PORT", 5000))

    )