from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load artifacts ─────────────────────────────────────────
MODEL_PATH   = 'models/best_model.pkl'
FEAT_PATH    = 'models/feature_cols.pkl'
SCALER_PATH  = 'models/scaler.pkl'

model        = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEAT_PATH)
scaler       = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# ── Preprocessing helper (mirrors model.py) ───────────────
def encode_input(form):
    """Convert raw form values → numeric feature vector."""
    import math

    gender        = 1 if form['gender'] == 'Male' else 0
    married       = 1 if form['married'] == 'Yes' else 0
    dep_map       = {'0': 0, '1': 1, '2': 2, '3+': 3}
    dependents    = dep_map.get(form['dependents'], 0)
    education     = 0 if form['education'] == 'Graduate' else 1
    self_employed = 1 if form['self_employed'] == 'Yes' else 0

    applicant_income   = float(form['applicant_income'])
    coapplicant_income = float(form['coapplicant_income'])
    loan_amount        = float(form['loan_amount'])
    loan_term          = float(form['loan_amount_term'])
    credit_history     = float(form['credit_history'])
    prop_map           = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    property_area      = prop_map.get(form['property_area'], 1)

    total_income          = applicant_income + coapplicant_income
    log_total_income      = math.log1p(total_income)
    log_loan_amount       = math.log1p(loan_amount)
    emi                   = loan_amount / (loan_term if loan_term > 0 else 1)
    balance_income        = total_income - (emi * 1000)
    income_to_loan_ratio  = total_income / (loan_amount + 1)
    debt_to_income        = (loan_amount * 1000) / (total_income + 1)
    risk_score            = (
        credit_history * 3 +
        (1 if income_to_loan_ratio > 10 else 0) * 2 +
        (1 if debt_to_income < 5 else 0) * 2
    )

    row = [
        gender, married, dependents, education, self_employed,
        log_total_income, log_loan_amount, loan_term, credit_history,
        property_area, emi, balance_income, income_to_loan_ratio,
        debt_to_income, risk_score
    ]
    return np.array(row).reshape(1, -1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X = encode_input(data)

        if scaler is not None:
            X = scaler.transform(X)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        return jsonify({
            'approved': bool(prediction),
            'probability': round(float(probability) * 100, 1),
            'status': 'Approved ✅' if prediction else 'Rejected ❌',
            'risk': 'Low' if probability > 0.75 else ('Medium' if probability > 0.5 else 'High')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
