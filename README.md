# Credit Risk Modelling

CreditLens is an end-to-end machine learning system for predicting **loan default probability** using applicant financial and demographic data. The project demonstrates a production style ML pipeline including preprocessing, feature engineering, hyperparameter tuning, probability calibration, threshold optimization, API deployment, and an interactive web interface.

---

## 📌 Problem Statement

Credit risk assessment is critical in financial services. Manual loan evaluation is inconsistent and inefficient.  

This project aims to:

- Analyze relationships between borrower attributes and loan default  
- Build a machine learning model to **predict probability of default**  
- Deploy the model via a **REST API** with a real-time frontend  

Loan default is treated as a binary classification task:

- `1` → Default  
- `0` → No Default  

---

## 🧰 Technology Stack

**Machine Learning**
- Python
- XGBoost
- Scikit-learn
- Pandas, NumPy
- SHAP (Explainability)
- SciPy (Drift Detection)
- Joblib (Model Serialization)

**Backend**
- Flask
- Flask-CORS

**Frontend**
- HTML, CSS, JavaScript

---

## 📂 Dataset

- Tabular loan applicant dataset (CSV)
- Target: `loan_status` (0 = No Default, 1 = Default)
- 13 raw financial & demographic features

Preprocessing includes:
- Median imputation (numeric)
- Mode imputation (categorical)
- 99.5th percentile outlier capping (training-set derived)
- Encoding (ordinal + label encoding)

---

## 🔍 Feature Engineering

Six derived financial ratios improve predictive power:

- `income_to_loan_ratio`
- `debt_burden`
- `credit_per_history`
- `annual_loan_payment`
- `payment_to_income`
- `emp_to_age_ratio`

These engineered features enhance nonlinear separability and model discrimination.

---

## ⚙️ Data Splitting Strategy (Leakage-Safe)

| Split | Purpose | Size |
|--------|----------|------|
| Train | Model training | 60% |
| Val-ES | Early stopping | 15% |
| Val-Cal | Probability calibration + threshold tuning | 12.5% |
| Test | Final evaluation | 20% |

This 4-way split ensures **no data leakage** across optimization stages.

---

## 🤖 Machine Learning Pipeline

**Primary Model:** XGBoost Classifier  
- RandomizedSearchCV (30 iterations)  
- 5-fold stratified cross-validation  
- Optimized for ROC-AUC  
- L1 & L2 regularization  
- Early stopping (50 rounds)  

**Calibration Model:** Logistic Regression  
- Platt Scaling on `val_cal` split  

---

## 📊 Model Performance (Held-Out Test Set)

| Metric | Value |
|--------|--------|
| Accuracy | **93.20%** |
| ROC-AUC | **0.9782** |
| F1-Score | **0.8452** |
| KS Statistic | **0.8286** |
| AUC Gap | **0.0213** (Low Overfitting) |

### Confusion Matrix

```
[[6717   283]
 [ 329  1671]]
```

✔ Strong class separation  
✔ Minimal overfitting (AUC gap < 0.03)  
✔ Balanced precision-recall on minority default class  

---

## 🎯 Probability Calibration & Threshold Optimization

- Raw XGBoost probabilities calibrated using **Platt Scaling**
- Decision threshold optimized for **maximum F1-score**
- Threshold selected on calibration set only (never test set)
- Risk levels mapped to interpretable lending decisions
- Credit score generated on 300–850 scale

---

## 🖥 Full-Stack Integration

- Interactive frontend dashboard
- Live model status indicator
- Animated risk visualization
- Auto loan-to-income calculation
- Offline fallback demo mode

---

## 🎯 Key Learnings

- Production-style ML validation workflow
- Probability calibration for risk modeling
- Overfitting detection via AUC gap
- KS statistic for financial model evaluation
- Drift detection using KS test
- REST API deployment with Flask
- Full-stack ML system integration

---

## ⚠️ Limitations

- Static dataset (no automated retraining pipeline)
- In-memory prediction logging (not persistent)
- No authentication layer for API
- Development Flask server (not production WSGI)
- No fairness/bias auditing
- SHAP computed on uncalibrated model (ranking valid, magnitudes not calibrated)
- Frontend fallback uses heuristic
