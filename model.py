import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_NUMERIC = [
    "person_age", "person_income", "person_emp_exp",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score",
]

RAW_CATEGORICAL = [
    "person_gender", "person_education", "person_home_ownership",
    "loan_intent", "previous_loan_defaults_on_file",
]

ENGINEERED = [
    "income_to_loan_ratio", "debt_burden", "credit_per_history",
    "annual_loan_payment", "payment_to_income", "emp_to_age_ratio",
]

ORDINAL_MAPS = {
    "person_education": {
        "High School": 0, "Associate": 1, "Bachelor": 2,
        "Master": 3, "Doctorate": 4,
    }
}

NOMINAL_FEATURES = [c for c in RAW_CATEGORICAL if c not in ORDINAL_MAPS]
ALL_FEATURES = RAW_NUMERIC + ENGINEERED + RAW_CATEGORICAL
TARGET = "loan_status"

print("=" * 60)
print("  CREDIT RISK MODEL — FINAL TRAINING")
print("=" * 60)

data = pd.read_csv(os.path.join(BASE_DIR, "loan_data.csv"))
data = data.dropna(subset=[TARGET])
data[TARGET] = data[TARGET].astype(int)

print(f"\nRows: {len(data):,} | Default rate: {data[TARGET].mean():.2%}")

X_raw = data.drop(columns=[TARGET])
y = data[TARGET].values

X_trainval, X_test_raw, y_trainval, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

X_train_raw, X_remaining, y_train, y_remaining = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

X_val_es_raw, X_val_cal_raw, y_val_es, y_val_cal = train_test_split(
    X_remaining, y_remaining, test_size=0.50, random_state=42, stratify=y_remaining
)

print(f"Train: {len(X_train_raw):,} | Val-ES: {len(X_val_es_raw):,} | Val-Cal: {len(X_val_cal_raw):,} | Test: {len(X_test_raw):,}")


drift_metrics = {}
try:
    from scipy.stats import ks_2samp
    for col in RAW_NUMERIC:
        stat = ks_2samp(X_train_raw[col].dropna(), X_test_raw[col].dropna()).statistic
        drift_metrics[col] = round(float(stat), 4)
    drifted = {k: v for k, v in drift_metrics.items() if v > 0.1}
    if drifted:
        print(f"\nDrift warnings: {drifted}")
    else:
        print("No significant feature drift detected.")
except ImportError:
    pass


outlier_caps = {
    "person_age": [18, 100],
    "person_emp_exp": [0, 60],
    "person_income": [0, float(X_train_raw["person_income"].quantile(0.995))],
    "loan_amnt": [1, float(X_train_raw["loan_amnt"].quantile(0.995))],
}


def apply_caps(df):
    df = df.copy()
    for col, (lo, hi) in outlier_caps.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def engineer(df):
    df = df.copy()
    safe_loan = df["loan_amnt"].replace(0, np.nan)
    df["income_to_loan_ratio"] = df["person_income"] / safe_loan
    df["debt_burden"] = df["loan_int_rate"] * df["loan_percent_income"]
    df["credit_per_history"] = df["credit_score"] / (df["cb_person_cred_hist_length"] + 1)
    df["annual_loan_payment"] = df["loan_amnt"] * df["loan_int_rate"] / 100
    df["payment_to_income"] = df["annual_loan_payment"] / (df["person_income"] + 1)
    df["emp_to_age_ratio"] = df["person_emp_exp"] / (df["person_age"] + 1)
    for col in ENGINEERED:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


splits = {
    "train":   (X_train_raw,   y_train),
    "val_es":  (X_val_es_raw,  y_val_es),
    "val_cal": (X_val_cal_raw, y_val_cal),
    "test":    (X_test_raw,    y_test),
}

processed = {}
for name, (xdf, yarr) in splits.items():
    processed[name] = (apply_caps(xdf), yarr)

for name in processed:
    xdf, yarr = processed[name]
    processed[name] = (engineer(xdf), yarr)


fill_values = {}
train_df = processed["train"][0]
for col in RAW_NUMERIC + ENGINEERED:
    if col in train_df.columns:
        fill_values[col] = float(train_df[col].median())
for col in RAW_CATEGORICAL:
    if col in train_df.columns:
        fill_values[col] = str(train_df[col].mode()[0])

for name in processed:
    xdf, yarr = processed[name]
    xdf = xdf.copy()
    for col, val in fill_values.items():
        if col in xdf.columns:
            xdf[col] = xdf[col].fillna(val)
    processed[name] = (xdf, yarr)


for name in processed:
    xdf, yarr = processed[name]
    xdf = xdf.copy()
    for col in RAW_CATEGORICAL:
        xdf[col] = xdf[col].astype(str).str.strip()
    processed[name] = (xdf, yarr)

for name in processed:
    xdf, yarr = processed[name]
    xdf = xdf.copy()
    for col, mapping in ORDINAL_MAPS.items():
        xdf[col] = xdf[col].map(mapping).fillna(0).astype(int)
    processed[name] = (xdf, yarr)


label_encoders = {}
train_df = processed["train"][0]
for col in NOMINAL_FEATURES:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    known = set(le.classes_)
    for name in ["val_es", "val_cal", "test"]:
        xdf, yarr = processed[name]
        xdf = xdf.copy()
        xdf[col] = xdf[col].apply(lambda x, k=known, d=le.classes_[0]: x if x in k else d)
        xdf[col] = le.transform(xdf[col])
        processed[name] = (xdf, yarr)
    label_encoders[col] = le
processed["train"] = (train_df, processed["train"][1])


for name in processed:
    xdf, yarr = processed[name]
    processed[name] = (xdf[ALL_FEATURES], yarr)

X_train,   y_train   = processed["train"]
X_val_es,  y_val_es  = processed["val_es"]
X_val_cal, y_val_cal = processed["val_cal"]
X_test,    y_test    = processed["test"]


scale_weight = float((y_train == 0).sum() / (y_train == 1).sum())

base_model = XGBClassifier(
    n_estimators=300,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

param_grid = {
    "learning_rate":    [0.01, 0.02, 0.03, 0.05, 0.08],
    "max_depth":        [4, 5, 6, 7],
    "min_child_weight": [1, 2, 3],
    "subsample":        [0.8, 0.85, 0.9],
    "colsample_bytree": [0.8, 0.85, 0.9],
    "gamma":            [0, 0.05, 0.1],
    "reg_alpha":        [0, 0.05, 0.1],
    "reg_lambda":       [0.5, 1.0, 1.5],
}

search = RandomizedSearchCV(
    base_model,
    param_grid,
    n_iter=30,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

print("\nSearching hyperparameters (n_iter=30, n_estimators=300 per fold)...")
search.fit(X_train, y_train)
best_params = search.best_params_
print(f"Best CV AUC: {search.best_score_:.4f}")


print("Training final model (n_estimators=3000, early_stopping=50)...")

final_model = XGBClassifier(
    **best_params,
    n_estimators=3000,
    early_stopping_rounds=50,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val_es, y_val_es)],
    verbose=False,
)

if (hasattr(final_model, "best_iteration")
        and final_model.best_iteration is not None
        and final_model.best_iteration > 0):
    trees_used = final_model.best_iteration + 1
    print(f"Early stopped at {trees_used} trees (saved {3000 - trees_used})")
else:
    trees_used = 3000
    print(f"Used all {trees_used} trees")

print("Calibrating probabilities (Platt scaling on val_cal)...")

val_cal_scores = final_model.predict_proba(X_val_cal)[:, 1].reshape(-1, 1)
platt = LogisticRegression(C=1.0, solver="lbfgs")
platt.fit(val_cal_scores, y_val_cal)


def calibrated_proba(X):
    """Return calibrated default probabilities for feature matrix X."""
    raw = final_model.predict_proba(X)[:, 1].reshape(-1, 1)
    return platt.predict_proba(raw)[:, 1]

val_cal_probs = calibrated_proba(X_val_cal)

best_f1 = 0
best_threshold = 0.5
for t in np.arange(0.05, 0.96, 0.005):
    preds = (val_cal_probs >= t).astype(int)
    f1 = f1_score(y_val_cal, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = round(float(t), 3)

print(f"Threshold: {best_threshold} | Cal-F1: {best_f1:.4f}")


train_probs = calibrated_proba(X_train)
test_probs  = calibrated_proba(X_test)
test_preds  = (test_probs >= best_threshold).astype(int)

train_auc = roc_auc_score(y_train, train_probs)
test_auc  = roc_auc_score(y_test,  test_probs)
test_acc  = accuracy_score(y_test,  test_preds)
test_f1   = f1_score(y_test,        test_preds)
auc_gap   = train_auc - test_auc

fpr, tpr, _ = roc_curve(y_test, test_probs)
ks_stat = float(max(tpr - fpr))

print(f"\n{'='*50}")
print(f"  TEST RESULTS (held-out, never seen)")
print(f"{'='*50}")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  ROC-AUC:   {test_auc:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  AUC Gap:   {auc_gap:.4f}  {'OK' if auc_gap < 0.03 else 'OVERFIT WARNING'}")
print(f"  KS Stat:   {ks_stat:.4f}")
print(f"{'='*50}")
print(f"\n{classification_report(y_test, test_preds)}")
print(confusion_matrix(y_test, test_preds))

shap_importance = {}
try:
    import shap
    explainer = shap.TreeExplainer(final_model)
    sample = X_train.sample(min(1000, len(X_train)), random_state=42)
    sv = explainer.shap_values(sample)
    mean_abs = np.abs(sv).mean(axis=0)
    shap_importance = dict(
        sorted(
            [(f, round(float(v), 4)) for f, v in zip(ALL_FEATURES, mean_abs)],
            key=lambda x: x[1], reverse=True,
        )
    )
    print("\nSHAP Top 8:")
    for f, v in list(shap_importance.items())[:8]:
        print(f"  {f:30s} {v:.4f}")
except ImportError:
    print("\nshap not installed — skipping SHAP explainability.")

imp = pd.Series(final_model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
print("\nGain Top 8:")
for f, v in imp.head(8).items():
    print(f"  {f:30s} {v:.4f}")

joblib.dump(
    {"model": final_model, "platt": platt},
    os.path.join(BASE_DIR, "credit_model.joblib"),
    compress=3,
)

joblib.dump(
    {"label_encoders": label_encoders, "ordinal_maps": ORDINAL_MAPS},
    os.path.join(BASE_DIR, "encoders_meta.joblib"),
    compress=3,
)

safe_params = {}
for k, v in best_params.items():
    if isinstance(v, np.integer):
        safe_params[k] = int(v)
    elif isinstance(v, np.floating):
        safe_params[k] = float(v)
    else:
        safe_params[k] = v

meta = {
    "threshold": best_threshold,
    "feature_order": ALL_FEATURES,
    "raw_numeric": RAW_NUMERIC,
    "raw_categorical": RAW_CATEGORICAL,
    "engineered": ENGINEERED,
    "ordinal_maps": ORDINAL_MAPS,
    "nominal_features": NOMINAL_FEATURES,
    "fill_values": fill_values,
    "outlier_caps": {k: [float(lo), float(hi)] for k, (lo, hi) in outlier_caps.items()},
    "best_params": safe_params,
    "metrics": {
        "cv_roc_auc":             round(float(search.best_score_), 4),
        "train_auc":              round(float(train_auc), 4),
        "test_accuracy":          round(float(test_acc), 4),
        "test_roc_auc":           round(float(test_auc), 4),
        "test_f1":                round(float(test_f1), 4),
        "auc_gap":                round(float(auc_gap), 4),
        "ks_statistic":           round(float(ks_stat), 4),
        "best_threshold":         best_threshold,
        "trees_used":             int(trees_used),
        "search_n_estimators":    300,
        "final_n_estimators_cap": 3000,
    },
    "drift_ks":   drift_metrics,
    "shap_top10": dict(list(shap_importance.items())[:10]),
    "notes": {
        "calibration": "Manual Platt scaling via LogisticRegression on val_cal scores. "
                       "Replaces CalibratedClassifierCV(cv='prefit') removed in sklearn 1.4+.",
        "early_stopping": "early_stopping_rounds set in XGBClassifier constructor (XGBoost 2.x API).",
        "drift": "KS computed on raw inputs before capping/engineering.",
        "shap": "Computed on uncalibrated model; rankings identical post-calibration (monotonic transform).",
    },
}

with open(os.path.join(BASE_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved: credit_model.joblib  (model + platt scaler)")
print(f"Saved: encoders_meta.joblib (label encoders + ordinal maps)")
print(f"Saved: model_meta.json      (all inference metadata)")
