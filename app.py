from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import traceback


app = Flask(__name__)
CORS(app)
model= joblib.load('models/diabetes_svm_pipeline_raw.joblib')
model_pima = joblib.load('models/PIMA_diabetes_model.joblib')
THRESHOLD = 0.40

@app.route("/predict_diabetes", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        df = pd.DataFrame([data])
        prob = float(model.predict_proba(df)[:, 1][0])
        pred = int(prob >= THRESHOLD)


        return jsonify({
            "prediction": pred,
            "probability": prob,
            "threshold_used": THRESHOLD
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_diabetes_pima", methods=["POST"])
def predict_pima():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        df = pd.DataFrame([data])

        prob = float(model_pima.predict_proba(df)[:, 1][0])
        pred = int(prob >= THRESHOLD)
        return jsonify({
            "prediction": pred,
            "probability": prob,
            "threshold_used": THRESHOLD
        }), 200




    except Exception as e:

        traceback.print_exc()  # full stack trace in your terminal

        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
