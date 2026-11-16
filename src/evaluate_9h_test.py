
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ==================================================
# 1. PROJECT PATH CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

# ==================================================
# 2. LOAD THE 9-HOUR MODEL MODULE
# ==================================================

MODEL_MODULE_PATH = SRC_ROOT / "early_warning_model_9h.py"

if not MODEL_MODULE_PATH.exists():
    raise FileNotFoundError(
        f"Could not find model module: {MODEL_MODULE_PATH}"
    )

spec = importlib.util.spec_from_file_location(
    "early_warning_model_9h",
    MODEL_MODULE_PATH
)

if spec is None or spec.loader is None:
    raise ImportError("Unable to load the 9-hour model module.")

early_warning_model_9h = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = early_warning_model_9h
spec.loader.exec_module(early_warning_model_9h)

extract_9h_features = early_warning_model_9h.extract_9h_features
split_9h_dataset = early_warning_model_9h.split_9h_dataset

# ==================================================
# 3. LOAD THE 9-HOUR DATASET
# ==================================================

print("\nLoading 9-hour dataset...")

X, y, meta_df, feature_names = extract_9h_features()

print("Dataset loaded successfully.")
print("Feature matrix shape:", X.shape)

# ==================================================
# 4. CHRONOLOGICAL DATA SPLIT
# ==================================================

print("\nCreating chronological data splits...")

(
    X_train, y_train, meta_train,
    X_val, y_val, meta_val,
    X_test, y_test, meta_test
) = split_9h_dataset(X, y, meta_df)


print("\nDataset distribution:")
print("Training samples  :", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples      :", len(X_test))

print("\nTest class distribution:")

class_names = {
    0: "Weak",
    1: "Moderate",
    2: "Intense+"
}

test_distribution = pd.Series(y_test).value_counts().sort_index()

for class_id in [0, 1, 2]:
    print(
        f"{class_names[class_id]}: "
        f"{test_distribution.get(class_id, 0)}"
    )

# ==================================================
# 5. LOAD THE SAVED REBUILT MODEL AND SCALER
# ==================================================

print("\nLoading saved 9-hour model and scaler...")

model_path = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "early_warning_model_9h_rebuilt.joblib"
)

scaler_path = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "early_warning_scaler_9h_rebuilt.joblib"
)

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not scaler_path.exists():
    raise FileNotFoundError(f"Scaler not found: {scaler_path}")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Model loaded successfully.")
print("Scaler loaded successfully.")

# ==================================================
# 6. SCALE THE TEST DATA
# ==================================================

print("\nScaling test data...")

X_test_scaled = scaler.transform(X_test)

# ==================================================
# 7. GENERATE PREDICTIONS
# ==================================================

print("\nGenerating test predictions...")

y_pred = model.predict(X_test_scaled)

# ==================================================
# 8. CALCULATE PERFORMANCE METRICS
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 55)
print("       9-HOUR MODEL: FINAL TEST RESULTS")
print("=" * 55)

print(f"\nTest samples : {len(y_test)}")
print(f"Accuracy     : {accuracy * 100:.2f}%")
print(f"Macro F1     : {macro_f1 * 100:.2f}%")
print(f"Weighted F1  : {weighted_f1 * 100:.2f}%")

# ==================================================
# 9. CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=["Weak", "Moderate", "Intense+"],
    zero_division=0
)

print(report)

# ==================================================
# 10. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print("\nConfusion Matrix:")
print(cm)

# ==================================================
# 11. SAVE TEST RESULTS
# ==================================================

output_path = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "early_warning_summary_9h_test.json"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

results = {
    "model": "9-hour rebuilt model",
    "test_period": "2016-2020",
    "test_samples": int(len(y_test)),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "classification_report": classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Weak", "Moderate", "Intense+"],
        output_dict=True,
        zero_division=0
    ),
    "confusion_matrix": cm.tolist()
}

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("\n" + "=" * 55)
print("TEST EVALUATION COMPLETED")
print("=" * 55)

print(f"\nResults saved to:\n{output_path}")
