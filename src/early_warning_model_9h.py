import json
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
"""
9-Hour Early Warning Model - Feature Extraction
================================================

Extracts Dst observations D0 to D9 (10 hourly readings)
for each storm and calculates six additional features.

Total features per storm: 16

This module does not train a model. It only prepares and
validates the 9-hour dataset.

The existing 6-hour model is not modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Project directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Input files
CLEAN_DST_PATH = (
    ROOT_DIR / "data" / "processed" / "dst_hourly_clean.csv"
)

CATALOGUE_PATH = (
    ROOT_DIR / "data" / "catalogue" / "storm_catalogue.csv"
)


def map_intensity_class(min_dst):
    """
    Map minimum Dst to the existing three intensity classes.

    Class 0: Weak       (Min Dst > -50 nT)
    Class 1: Moderate   (-100 < Min Dst <= -50 nT)
    Class 2: Intense+   (Min Dst <= -100 nT)
    """
    if min_dst > -50.0:
        return 0
    elif min_dst > -100.0:
        return 1
    else:
        return 2


CLASS_NAMES = [
    "Weak",
    "Moderate",
    "Intense+"
]


def extract_9h_features(
    clean_dst_path=CLEAN_DST_PATH,
    catalogue_path=CATALOGUE_PATH
):
    """
    Extract exactly 10 consecutive hourly Dst observations:

        D0, D1, D2, ..., D9

    Incomplete or non-contiguous windows are skipped.
    No artificial padding is used.

    Returns
    -------
    X : numpy array
        Feature matrix with 16 features per storm.

    y : numpy array
        Storm intensity labels.

    meta_df : pandas DataFrame
        Storm metadata, including lead time from hour 9.

    feature_names : list
        Names of all 16 features.
    """

    print("Loading cleaned Dst data...")
    dst_df = pd.read_csv(clean_dst_path)

    required_dst_columns = {"timestamp_utc", "dst_nt"}

    if not required_dst_columns.issubset(dst_df.columns):
        raise ValueError(
            "Dst CSV must contain timestamp_utc and dst_nt columns."
        )

    dst_df["timestamp_utc"] = pd.to_datetime(
        dst_df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    dst_df["dst_nt"] = pd.to_numeric(
        dst_df["dst_nt"],
        errors="coerce"
    )

    dst_df = dst_df.dropna(
        subset=["timestamp_utc", "dst_nt"]
    )

    dst_df = dst_df.sort_values("timestamp_utc")

    # Remove duplicate timestamps, if any.
    dst_df = dst_df.drop_duplicates(
        subset="timestamp_utc",
        keep="last"
    )

    dst_series = (
        dst_df.set_index("timestamp_utc")["dst_nt"]
        .sort_index()
    )

    print("Loading storm catalogue...")
    cat_df = pd.read_csv(catalogue_path)

    required_catalogue_columns = {
        "storm_id",
        "start_time",
        "peak_time",
        "min_dst_nt",
        "main_phase_hours"
    }

    if not required_catalogue_columns.issubset(cat_df.columns):
        raise ValueError(
            "Storm catalogue is missing one or more required columns: "
            + str(required_catalogue_columns)
        )

    cat_df["start_time"] = pd.to_datetime(
        cat_df["start_time"],
        utc=True,
        errors="coerce"
    )

    cat_df["peak_time"] = pd.to_datetime(
        cat_df["peak_time"],
        utc=True,
        errors="coerce"
    )

    cat_df["min_dst_nt"] = pd.to_numeric(
        cat_df["min_dst_nt"],
        errors="coerce"
    )

    cat_df = cat_df.dropna(
        subset=["start_time", "peak_time", "min_dst_nt"]
    )

    features = []
    labels = []
    storm_metadata = []

    skipped_incomplete = 0

    print(
        f"Extracting 9-hour trajectories from "
        f"{len(cat_df)} catalogue storms..."
    )

    for _, row in cat_df.iterrows():

        t0 = row["start_time"]

        # Decision time: 9 hours after storm onset.
        t9 = t0 + pd.Timedelta(hours=9)

        # Expected timestamps: D0 through D9.
        expected_times = pd.date_range(
            start=t0,
            periods=10,
            freq="h"
        )

        # Select only the exact expected hourly timestamps.
        vals = dst_series.reindex(expected_times)

        # Require all 10 readings to be present and finite.
        if vals.isna().any():
            skipped_incomplete += 1
            continue

        values = vals.to_numpy(dtype=float)

        if len(values) != 10 or not np.isfinite(values).all():
            skipped_incomplete += 1
            continue

        # Raw Dst trajectory.
        d0 = values[0]
        d4 = values[4]
        d5 = values[5]
        d9 = values[9]

        # Six engineered features.
        min_9h = np.min(values)

        drop_9h = d0 - min_9h

        slope_9h = (d9 - d0) / 9.0

        # Compare the two four-hour slopes:
        # D0 to D4 and D5 to D9.
        accel_9h = (
            (d9 - d5) / 4.0
            -
            (d4 - d0) / 4.0
        )

        volatility_9h = np.std(values)

        mean_9h = np.mean(values)

        # Total feature vector: 10 raw + 6 engineered.
        feature_vector = list(values) + [
            min_9h,
            drop_9h,
            slope_9h,
            accel_9h,
            volatility_9h,
            mean_9h
        ]

        # Final storm intensity class from catalogue minimum Dst.
        label = map_intensity_class(row["min_dst_nt"])

        # Lead time measured from the hour-9 decision point.
        lead_time_hours = (
            row["peak_time"] - t9
        ).total_seconds() / 3600.0

        features.append(feature_vector)
        labels.append(label)

        storm_metadata.append({
            "storm_id": row["storm_id"],
            "start_time": row["start_time"],
            "peak_time": row["peak_time"],
            "min_dst_nt": row["min_dst_nt"],
            "main_phase_hours": row["main_phase_hours"],
            "lead_time_hours": lead_time_hours
        })

    feature_names = [
        "dst_0h",
        "dst_1h",
        "dst_2h",
        "dst_3h",
        "dst_4h",
        "dst_5h",
        "dst_6h",
        "dst_7h",
        "dst_8h",
        "dst_9h",
        "min_9h",
        "drop_9h",
        "slope_9h",
        "accel_9h",
        "volatility_9h",
        "mean_9h"
    ]

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)

    meta_df = pd.DataFrame(storm_metadata)

    print("\n" + "=" * 55)
    print("9-HOUR FEATURE EXTRACTION SUMMARY")
    print("=" * 55)

    print(f"Catalogue storms:          {len(cat_df)}")
    print(f"Successfully extracted:    {len(X)}")
    print(f"Skipped incomplete windows:{skipped_incomplete}")
    print(f"Number of features:        {len(feature_names)}")

    print("\nIntensity class distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == class_id))
        print(f"{class_name:12s}: {count}")

    print("\nLead-time summary:")
    if len(meta_df) > 0:
        print(
            "Mean lead time:",
            round(meta_df["lead_time_hours"].mean(), 2),
            "hours"
        )

        print(
            "Median lead time:",
            round(meta_df["lead_time_hours"].median(), 2),
            "hours"
        )

        print(
            "Positive lead time (%):",
            round(
                (meta_df["lead_time_hours"] > 0).mean() * 100,
                2
            )
        )

    print("\nFeature matrix shape:", X.shape)
    print("Label array shape:", y.shape)

    return X, y, meta_df, feature_names


def split_9h_dataset(X, y, meta_df):
    """
    Chronological splitting of the 9-hour dataset.

    Training:   1957-2009
    Validation: 2010-2015
    Testing:    2016-2020

    No random shuffling is used.
    """

    print("\nPreparing chronological datasets...")

    # Define chronological periods.
    train_mask = (
        meta_df["start_time"] < pd.Timestamp("2010-01-01", tz="UTC")
    )

    val_mask = (
        (meta_df["start_time"] >= pd.Timestamp("2010-01-01", tz="UTC"))
        &
        (meta_df["start_time"] < pd.Timestamp("2016-01-01", tz="UTC"))
    )

    test_mask = (
        (meta_df["start_time"] >= pd.Timestamp("2016-01-01", tz="UTC"))
        &
        (meta_df["start_time"] < pd.Timestamp("2021-01-01", tz="UTC"))
    )

    # Extract datasets.
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    meta_train = meta_df.loc[train_mask].reset_index(drop=True)
    meta_val = meta_df.loc[val_mask].reset_index(drop=True)
    meta_test = meta_df.loc[test_mask].reset_index(drop=True)

    # Print dataset sizes.
    print("\n" + "=" * 55)
    print("9-HOUR CHRONOLOGICAL DATASET SPLIT")
    print("=" * 55)

    print(f"Training storms:   {len(y_train)} (1957-2009)")
    print(f"Validation storms: {len(y_val)} (2010-2015)")
    print(f"Testing storms:    {len(y_test)} (2016-2020)")

    print("\nClass distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):

        train_count = int(np.sum(y_train == class_id))
        val_count = int(np.sum(y_val == class_id))
        test_count = int(np.sum(y_test == class_id))

        print(f"\n{class_name}")
        print(f"  Training:   {train_count}")
        print(f"  Validation: {val_count}")
        print(f"  Testing:    {test_count}")

    # Check dataset dimensions.
    print("\nFeature dimensions:")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    # Check that all three datasets are non-empty.
    if min(len(y_train), len(y_val), len(y_test)) == 0:
        raise ValueError(
            "One or more chronological datasets are empty."
        )

    # Confirm that all datasets have 16 features.
    if X_train.shape[1] != 16:
        raise ValueError("Expected exactly 16 input features.")

    print("\nChronological splitting completed successfully.")

    return (
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        X_test, y_test, meta_test
    )


def train_9h_model(
    X_train, y_train,
    X_val, y_val,
    feature_names
):
    """
    Train a separate 9-hour Gradient Boosting model.

    Training data is used for fitting.
    Validation data is used for model evaluation.
    Test data is not accessed in this function.
    """

    print("\n" + "=" * 60)
    print("TRAINING THE 9-HOUR EARLY WARNING MODEL")
    print("=" * 60)

    # 1. Feature scaling
    print("\nStep 1: Scaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Training features scaled.")
    print("Validation features scaled.")

    # 2. Initialize Gradient Boosting classifier
    print("\nStep 2: Initializing Gradient Boosting model...")

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )

    # 3. Train using training data only
    print("\nStep 3: Fitting the model...")

    model.fit(X_train_scaled, y_train)

    print("Model training completed.")

    # 4. Validation predictions
    print("\nStep 4: Evaluating on validation data...")

    y_val_pred = model.predict(X_val_scaled)

    # 5. Calculate validation metrics
    accuracy = accuracy_score(y_val, y_val_pred)

    macro_f1 = f1_score(
        y_val,
        y_val_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        y_val_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_val,
        y_val_pred,
        labels=[0, 1, 2]
    )

    report = classification_report(
        y_val,
        y_val_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("9-HOUR VALIDATION RESULTS")
    print("=" * 60)

    print(f"Validation Accuracy:  {accuracy * 100:.2f}%")
    print(f"Validation Macro F1:  {macro_f1 * 100:.2f}%")
    print(f"Validation Weighted F1: {weighted_f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            y_val_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(cm)

    # 6. Feature importance
    print("\nTop 10 Important Features:")

    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    for rank, idx in enumerate(sorted_indices[:10], start=1):
        print(
            f"{rank}. {feature_names[idx]}: "
            f"{importances[idx] * 100:.2f}%"
        )

    # 7. Save separate 9-hour model artifacts
    model_dir = ROOT_DIR / "outputs" / "models"
    metrics_dir = ROOT_DIR / "outputs" / "metrics"

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "early_warning_model_9h_rebuilt.joblib"
    scaler_path = model_dir / "early_warning_scaler_9h_rebuilt.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save validation metrics
    summary = {
        "model": "Gradient Boosting",
        "observation_window_hours": 9,
        "number_of_features": len(feature_names),
        "training_storms": int(len(y_train)),
        "validation_storms": int(len(y_val)),
        "validation_accuracy": float(accuracy),
        "validation_macro_f1": float(macro_f1),
        "validation_weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importance": [
            {
                "feature": feature_names[i],
                "importance": float(importances[i])
            }
            for i in sorted_indices
        ]
    }

    metrics_path = metrics_dir / "early_warning_summary_9h_rebuilt.json"

    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved 9-hour model:", model_path)
    print("Saved 9-hour scaler:", scaler_path)
    print("Saved validation metrics:", metrics_path)

    print("\n9-hour baseline training completed.")

    return model, scaler, summary



if __name__ == "__main__":

    # Step 1: Extract features.
    X, y, meta_df, feature_names = extract_9h_features()

    # Step 2: Chronological splitting.
    (
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        X_test, y_test, meta_test
    ) = split_9h_dataset(X, y, meta_df)

    # Step 3: Train and validate the 9-hour model.
    model, scaler, summary = train_9h_model(
        X_train, y_train,
        X_val, y_val,
        feature_names
    )

    print("\n9-hour model pipeline completed.")

