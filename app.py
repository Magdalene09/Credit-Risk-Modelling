import json
import logging
import os
import traceback
from collections import deque

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)

# allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
# CORS(app, resources={r"/*": {"origins": allowed_origins}}) for production use
CORS(app, resources={r"/*": {"origins": "*"}})
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

bundle       = joblib.load(os.path.join(BASE_DIR, "credit_model.joblib"))
final_model  = bundle["model"]
platt        = bundle["platt"]

enc_meta       = joblib.load(os.path.join(BASE_DIR, "encoders_meta.joblib"))
label_encoders = enc_meta["label_encoders"]
ordinal_maps   = enc_meta["ordinal_maps"]

with open(os.path.join(BASE_DIR, "model_meta.json")) as f:
    meta = json.load(f)

threshold      = meta["threshold"]
feature_order  = meta["feature_order"]
fill_values    = meta["fill_values"]
caps           = meta["outlier_caps"]
engineered_cols = meta["engineered"]


assert set(feature_order) == set(meta["feature_order"]), "Feature mismatch between model and meta"
assert "metrics" in meta, "model_meta.json missing metrics block"

logger.info(
    f"Model loaded — Test AUC: {meta['metrics']['test_roc_auc']} | "
    f"KS: {meta['metrics']['ks_statistic']} | "
    f"Threshold: {threshold}"
)



prediction_log = deque(maxlen=1000)



REQUIRED = [
    "person_age", "person_gender", "person_education",
    "person_income", "person_emp_exp", "person_home_ownership",
    "loan_amnt", "loan_intent", "loan_int_rate",
    "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file",
]


def process_input(df):
    df = df.copy()

    
    for col, bounds in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(bounds[0], bounds[1])

    
    safe_loan = df["loan_amnt"].replace(0, np.nan)
    df["income_to_loan_ratio"] = df["person_income"] / safe_loan
    df["debt_burden"]          = df["loan_int_rate"] * df["loan_percent_income"]
    df["credit_per_history"]   = df["credit_score"] / (df["cb_person_cred_hist_length"] + 1)
    df["annual_loan_payment"]  = df["loan_amnt"] * df["loan_int_rate"] / 100
    df["payment_to_income"]    = df["annual_loan_payment"] / (df["person_income"] + 1)
    df["emp_to_age_ratio"]     = df["person_emp_exp"] / (df["person_age"] + 1)

    for col in engineered_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    
    for col in label_encoders.keys() | ordinal_maps.keys():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    
    for col, le in label_encoders.items():
        if col in df.columns:
            known = set(le.classes_)
            df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])

    for col in feature_order:
        if col not in df.columns:
            df[col] = fill_values.get(col, 0)

    return df[feature_order]


def calibrated_proba(X):
    raw = final_model.predict_proba(X)[:, 1].reshape(-1, 1)
    return platt.predict_proba(raw)[:, 1]



def get_risk_level(prob):
    if prob < 0.15:
        return "Very Low Risk : Approved"
    if prob < 0.35:
        return "Low Risk : Approved"
    if prob < 0.50:
        return "Moderate Risk : Manual Review"
    if prob < 0.70:
        return "High Risk : Caution"
    return "Very High Risk : Denied"


def probability_to_score(prob):
    return int(850 - (prob * 550))


def validate(raw):
    errors = []
    if raw["person_age"] < 18 or raw["person_age"] > 120:
        errors.append("Age must be between 18 and 120")
    if raw["person_income"] < 0:
        errors.append("Income cannot be negative")
    if raw["loan_amnt"] <= 0:
        errors.append("Loan amount must be greater than 0")
    if raw["credit_score"] < 300 or raw["credit_score"] > 850:
        errors.append("Credit score must be between 300 and 850")
    if raw["loan_int_rate"] <= 0 or raw["loan_int_rate"] > 50:
        errors.append("Interest rate must be between 0 and 50")
    if raw["loan_percent_income"] < 0:
        errors.append("Loan percent of income cannot be negative")
    if raw["person_emp_exp"] < 0:
        errors.append("Employment experience cannot be negative")
    if raw["cb_person_cred_hist_length"] < 0:
        errors.append("Credit history length cannot be negative")
    return errors


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "threshold": threshold,
        "features_count": len(feature_order),
        "metrics": meta["metrics"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        missing = [f for f in REQUIRED if f not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        
        try:
            raw = {
                "person_age":                     float(payload["person_age"]),
                "person_gender":                  str(payload["person_gender"]).strip(),
                "person_education":               str(payload["person_education"]).strip(),
                "person_income":                  float(payload["person_income"]),
                "person_emp_exp":                 float(payload["person_emp_exp"]),
                "person_home_ownership":          str(payload["person_home_ownership"]).strip(),
                "loan_amnt":                      float(payload["loan_amnt"]),
                "loan_intent":                    str(payload["loan_intent"]).strip(),
                "loan_int_rate":                  float(payload["loan_int_rate"]),
                "loan_percent_income":            float(payload["loan_percent_income"]),
                "cb_person_cred_hist_length":     float(payload["cb_person_cred_hist_length"]),
                "credit_score":                   float(payload["credit_score"]),
                "previous_loan_defaults_on_file": str(payload["previous_loan_defaults_on_file"]).strip(),
            }
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid data type: {str(e)}"}), 400

        
        errors = validate(raw)
        if errors:
            return jsonify({"error": " | ".join(errors)}), 400

        
        df          = pd.DataFrame([raw])
        df          = process_input(df)
        probability = float(calibrated_proba(df)[0])
        prediction  = int(probability >= threshold)
        score       = probability_to_score(probability)
        risk        = get_risk_level(probability)

        prediction_log.append(probability)
        logger.info(json.dumps({
            "probability":  round(probability, 4),
            "prediction":   prediction,
            "risk_level":   risk,
            "credit_score": score,
            "input_summary": {
                "person_age":   raw["person_age"],
                "loan_amnt":    raw["loan_amnt"],
                "credit_score": raw["credit_score"],
            },
        }))

        return jsonify({
            "probability":     round(probability, 4),
            "prediction":      prediction,
            "risk_level":      risk,
            "credit_score":    score,
            "threshold_used":  round(threshold, 3),
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error. Contact support."}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """Live prediction distribution — useful for detecting model drift."""
    if not prediction_log:
        return jsonify({"message": "No predictions yet"})
    arr = np.array(prediction_log)
    return jsonify({
        "n":                      len(arr),
        "mean_probability":       round(float(arr.mean()), 4),
        "default_rate_estimated": round(float((arr >= threshold).mean()), 4),
        "p25":                    round(float(np.percentile(arr, 25)), 4),
        "p75":                    round(float(np.percentile(arr, 75)), 4),
    })


if __name__ == "__main__":
    print(f"\nThreshold : {threshold}")
    print(f"Features  : {len(feature_order)}")
    print(f"Metrics   : {meta['metrics']}")
    print(f"Server    : http://127.0.0.1:5000\n")

    app.run(
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 5000)),
    )