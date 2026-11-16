
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "outputs/models/early_warning_model_9h_rebuilt.joblib"
)

SCALER_PATH = (
    ROOT / "outputs/models/early_warning_scaler_9h_rebuilt.joblib"
)

FEATURE_NAMES = [
    "dst_0h", "dst_1h", "dst_2h", "dst_3h", "dst_4h",
    "dst_5h", "dst_6h", "dst_7h", "dst_8h", "dst_9h",
    "min_9h", "drop_9h", "slope_9h", "accel_9h",
    "volatility_9h", "mean_9h"
]

CLASS_NAMES = {
    0: "Weak",
    1: "Moderate",
    2: "Intense+"
}


def predict_9h_intensity(live_df, onset_time):
    """
    Predict eventual storm intensity from ten consecutive
    hourly Dst observations beginning at the identified onset.

    This function does not detect storm onset itself.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not SCALER_PATH.exists():
        raise FileNotFoundError(SCALER_PATH)

    required = {"timestamp", "dst_nt"}

    if not required.issubset(live_df.columns):
        raise ValueError(
            "Live data must contain timestamp and dst_nt."
        )

    df = live_df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    df["dst_nt"] = pd.to_numeric(
        df["dst_nt"], errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "dst_nt"]
    )

    df = df.drop_duplicates(
        subset="timestamp", keep="last"
    ).set_index("timestamp").sort_index()

    onset_time = pd.Timestamp(onset_time)

    if onset_time.tzinfo is None:
        onset_time = onset_time.tz_localize("UTC")
    else:
        onset_time = onset_time.tz_convert("UTC")

    expected = pd.date_range(
        start=onset_time,
        periods=10,
        freq="h"
    )

    values = df["dst_nt"].reindex(expected)

    if values.isna().any():
        return {
            "status": "waiting",
            "message": "Waiting for ten complete hourly Dst observations.",
            "required_until": expected[-1]
        }

    x = values.to_numpy(dtype=float)

    if not np.isfinite(x).all():
        return {
            "status": "waiting",
            "message": "Invalid Dst values in observation window."
        }

    d0, d4, d5, d9 = x[0], x[4], x[5], x[9]

    features = np.array(
        list(x) + [
            np.min(x),
            d0 - np.min(x),
            (d9 - d0) / 9.0,
            (d9 - d5) / 4.0 - (d4 - d0) / 4.0,
            np.std(x),
            np.mean(x)
        ],
        dtype=float
    ).reshape(1, -1)

    if features.shape[1] != 16:
        raise ValueError("Expected exactly 16 features.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    scaled = scaler.transform(features)

    prediction = int(model.predict(scaled)[0])

    return {
        "status": "success",
        "prediction": CLASS_NAMES[prediction],
        "prediction_id": prediction,
        "onset_time": onset_time,
        "decision_time": expected[-1],
        "dst_values": x.tolist(),
        "features": features[0].tolist(),
        "model": "9-hour rebuilt model",
        "data_quality": "Complete ten-hour window"
    }
