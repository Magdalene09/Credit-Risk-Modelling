# Credit Risk Modelling

An end-to-end **credit risk assessment platform** that combines advanced machine learning, probability calibration, risk monitoring, and a REST API backend with an interactive web interface.

CreditLens predicts **loan default probability**, generates a **credit score (300–850 scale)**, and outputs an interpretable **risk decision recommendation** — designed to mirror real-world fintech risk pipelines.

---

# 🔎 Project Highlights

* ✅ XGBoost model with randomized hyperparameter tuning (5-fold CV)
* ✅ Early stopping using dedicated validation split
* ✅ Platt scaling for probability calibration
* ✅ F1-optimized decision threshold (calibration-set tuned)
* ✅ KS Statistic & AUC gap monitoring
* ✅ Feature drift detection (Kolmogorov–Smirnov test)
* ✅ SHAP explainability support
* ✅ REST API (Flask) for real-time inference
* ✅ Interactive frontend dashboard

---

# 🧠 Full ML Lifecycle Implementation

1. Data preprocessing & encoding
2. Feature engineering (derived financial ratios)
3. Stratified data splitting (4-way split to prevent leakage)
4. Hyperparameter search (RandomizedSearchCV)
5. Early stopping validation
6. Probability calibration (Platt scaling)
7. Threshold optimization (F1-score based)
8. Model serialization
9. REST API deployment
10. Frontend integration

---

# 🛠 Tech Stack

### Machine Learning

Python · XGBoost · Scikit-learn · Pandas · NumPy · SHAP · SciPy · Joblib

### Backend

Flask · Flask-CORS · REST APIs

### Frontend

HTML · CSS · JavaScript · Google Fonts

---

# 📊 Model Architecture

## Data Splits (Leakage-Safe Design)

| Split   | Purpose                        | Size  |
| ------- | ------------------------------ | ----- |
| Train   | Model fitting                  | 60%   |
| Val-ES  | Early stopping                 | 15%   |
| Val-Cal | Calibration + threshold tuning | 12.5% |
| Test    | Final evaluation               | 20%   |

---

## Feature Engineering

Six derived financial features enhance predictive power:

* Income-to-loan ratio
* Debt burden
* Credit-per-history ratio
* Annual loan payment
* Payment-to-income ratio
* Employment-to-age ratio

Outliers are capped using percentile bounds derived from training data.

---

## Hyperparameter Optimization

Randomized search (30 iterations, 5-fold stratified CV) optimizing ROC-AUC across:

`learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda`

---

## Probability Calibration

Raw boosted-tree probabilities are calibrated using **Platt Scaling** (logistic regression on held-out validation scores) to correct overconfidence.

---

# 📈 Model Performance (Held-Out Test Set)

Evaluation performed on a completely unseen test split (20%).

```
==================================================
TEST RESULTS (Held-Out, Never Seen)
==================================================
Accuracy:   0.9320
ROC-AUC:    0.9782
F1-Score:   0.8452
AUC Gap:    0.0213  ✅ (Low Overfitting)
KS Stat:    0.8286
==================================================
```

## Classification Report

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Non-Default (0) | 0.95      | 0.96   | 0.96     | 7000    |
| Default (1)     | 0.86      | 0.84   | 0.85     | 2000    |

Overall Accuracy: **93.2%**
Weighted F1-Score: **0.93**

## Confusion Matrix

```
[[6717   283]
 [ 329  1671]]
```

* True Negatives: 6717
* False Positives: 283
* False Negatives: 329
* True Positives: 1671

### Model Strengths

* Excellent separation power (**ROC-AUC = 0.9782**)
* Strong class discrimination (**KS = 0.8286**)
* Minimal overfitting (**AUC gap = 0.0213**)
* Balanced precision-recall performance on minority default class

This performance indicates a production-quality risk scoring model with strong generalization capability.

---

# 🖥 Frontend Features

* Live API status indicator
* Auto loan-to-income calculation
* Animated probability visualization
* Credit score ring display
* Risk classification breakdown
* Offline fallback simulation mode

---

# ⚠️ Limitations

* No authentication layer
* Not optimized for distributed production deployment
* Drift detection is feature-level only (not population-level modeling)
