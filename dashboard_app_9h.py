
"""
Geomagnetic Storm Monitoring and 9-Hour Early Warning Dashboard

Run from the project root:
    streamlit run dashboard_app_9h.py
"""

from pathlib import Path
import sys
import json
import platform

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

platform._wmi = None

# ------------------------------------------------------------
# TIME DISPLAY: UTC internally, IST on dashboard
# ------------------------------------------------------------

DISPLAY_TIMEZONE = "Asia/Kolkata"

def convert_to_ist(timestamp):
    """Convert a timestamp to Indian Standard Time for display."""
    return pd.to_datetime(
        timestamp, utc=True
    ).tz_convert(DISPLAY_TIMEZONE)

def format_ist(timestamp, fmt="%d %b %Y, %I:%M %p"):
    """Format a timestamp as Indian Standard Time."""
    return convert_to_ist(timestamp).strftime(fmt) + " IST"


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT))


CATALOGUE_PATH = ROOT / "data/catalogue/storm_catalogue.csv"
CNN_METRICS_PATH = ROOT / "outputs/metrics/metrics.json"
FIGURES_DIR = ROOT / "outputs/figures"

MODEL_9H_PATH = ROOT / "outputs/models/early_warning_model_9h_rebuilt.joblib"
SCALER_9H_PATH = ROOT / "outputs/models/early_warning_scaler_9h_rebuilt.joblib"

METRICS_9H_PATH = ROOT / "outputs/metrics/early_warning_summary_9h.json"
METRICS_9H_TEST_PATH = ROOT / "outputs/metrics/early_warning_summary_9h_test.json"


# ------------------------------------------------------------
# PROJECT MODULES — IMPORT AS PART OF THE SRC PACKAGE
# ------------------------------------------------------------

try:
    from src.live_dst_monitor import fetch_live_dst
    LIVE_ERROR = None
except Exception as e:
    fetch_live_dst = None
    LIVE_ERROR = str(e)

try:
    from src.storm_detection import Params, detect_storms
    STORM_ERROR = None
except Exception as e:
    Params = None
    detect_storms = None
    STORM_ERROR = str(e)

try:
    from src.live_prediction_9h import predict_9h_intensity
    PREDICTION_ERROR = None
except Exception as e:
    predict_9h_intensity = None
    PREDICTION_ERROR = str(e)


# ------------------------------------------------------------
# STREAMLIT CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Geomagnetic Storm AI Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

@st.cache_data
def load_catalogue():

    if not CATALOGUE_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CATALOGUE_PATH)

    for col in ["start_time", "peak_time", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                utc=True,
                errors="coerce"
            )

    return df


@st.cache_data
def load_cnn_metrics():

    if not CNN_METRICS_PATH.exists():
        return {}

    with open(CNN_METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_live_data():

    if fetch_live_dst is None:
        raise ImportError(LIVE_ERROR)

    return fetch_live_dst()


def prepare_live_data(df):

    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "dst_nt"])

    df = df.copy()

    if "timestamp" not in df.columns or "dst_nt" not in df.columns:
        raise ValueError(
            "Live data must contain timestamp and dst_nt columns."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df["dst_nt"] = pd.to_numeric(
        df["dst_nt"],
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp", "dst_nt"])

    df = df.sort_values("timestamp")

    df = df.drop_duplicates(
        subset="timestamp",
        keep="last"
    )

    return df.reset_index(drop=True)


def utc_timestamp(value):

    value = pd.Timestamp(value)

    if value.tzinfo is None:
        return value.tz_localize("UTC")

    return value.tz_convert("UTC")


# ------------------------------------------------------------
# STORM DETECTION
# ------------------------------------------------------------


def detect_live_storms(live_df):

    if Params is None or detect_storms is None:
        raise ImportError(
            f"Storm detection import failed: {STORM_ERROR}"
        )

    import yaml

    config_path = ROOT / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    params = Params.from_config(config)

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

    df = (
        df.dropna(subset=["timestamp", "dst_nt"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
    )

    series = (
        df.set_index("timestamp")["dst_nt"]
        .sort_index()
    )

    storms = detect_storms(series, params)

    return storms


def convert_storms_to_rows(storms):

    if isinstance(storms, pd.DataFrame):
        return [
            row.to_dict()
            for _, row in storms.iterrows()
        ]

    if isinstance(storms, (list, tuple)):
        return list(storms)

    if isinstance(storms, dict):

        for key in ["storms", "events", "storm_catalogue"]:

            if key in storms:
                return convert_storms_to_rows(storms[key])

    return []


# ------------------------------------------------------------
# LIVE NINE-HOUR PREDICTION
# ------------------------------------------------------------


def display_9h_predictions(live_df):

    st.subheader("Current Geomagnetic Storm Status")

    if live_df.empty:
        st.warning("No live Dst observations are available.")
        return

    if predict_9h_intensity is None:
        st.error("Nine-hour prediction module is unavailable.")
        st.code(PREDICTION_ERROR or "Import error")
        return

    # --------------------------------------------------------
    # 1. LATEST OBSERVED DST
    # --------------------------------------------------------

    df = live_df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    df["dst_nt"] = pd.to_numeric(
        df["dst_nt"], errors="coerce"
    )

    df = (
        df.dropna(subset=["timestamp", "dst_nt"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
    )

    if df.empty:
        st.warning("No valid Dst observations.")
        return

    latest = df.iloc[-1]
    latest_dst = float(latest["dst_nt"])
    latest_time = latest["timestamp"]

    age_hours = (
        pd.Timestamp.now(tz="UTC") - latest_time
    ).total_seconds() / 3600

    c1, c2, c3 = st.columns(3)

    c1.metric("Latest Dst", f"{latest_dst:.1f} nT")
    c2.metric(
        "Latest Observation",
        format_ist(latest_time)
    )
    c3.metric("Observation Age", f"{age_hours:.1f} hours")

    if age_hours > 3:
        st.warning(
            "The latest observation is stale. "
            "Treat the live assessment with caution."
        )

    # --------------------------------------------------------
    # 2. CURRENT OBSERVED DST CONDITION
    # --------------------------------------------------------

    st.subheader("Current Observed Dst Condition")

    if latest_dst <= -100:

        st.error(
            "INTENSE+ RANGE — latest observed Dst is "
            "at or below -100 nT."
        )

    elif latest_dst <= -50:

        st.warning(
            "MODERATE RANGE — latest observed Dst is "
            "between -50 and -100 nT."
        )

    elif latest_dst <= -30:

        st.warning(
            "WEAK STORM-LEVEL RANGE — latest observed Dst "
            "has crossed -30 nT."
        )

    else:

        st.success(
            "NO STORM THRESHOLD CROSSING — latest observed "
            "Dst is above -30 nT."
        )

    st.caption(
        "This describes the latest observation only. "
        "It is not the nine-hour AI forecast."
    )

    # --------------------------------------------------------
    # 3. DETECT STORMS USING THE PROJECT DETECTOR
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("Detected Storm Events")

    try:
        storms = detect_live_storms(df)

    except Exception as e:
        st.error("Storm detection failed.")
        st.exception(e)
        return

    if storms is None or storms.empty:

        st.info(
            "No storm events were detected in the currently "
            "available observations using the configured "
            "storm-detection rules."
        )
        return

    storms = storms.copy()

    storms["start_time"] = pd.to_datetime(
        storms["start_time"], utc=True, errors="coerce"
    )

    storms = (
        storms.dropna(subset=["start_time"])
        .sort_values("start_time", ascending=False)
    )

    st.write(f"Detected events: **{len(storms)}**")

    # --------------------------------------------------------
    # 4. DISPLAY EVENT STATUS AND NINE-HOUR PREDICTION
    # --------------------------------------------------------

    eligible_events = 0

    for _, storm in storms.iterrows():

        onset = storm["start_time"]

        quality_flag = str(
            storm.get("data_quality_flag", "")
        )

        storm_id = storm.get("storm_id", "Unknown event")

        # Do not use an onset truncated by the data boundary.
        if "start_at_data_edge" in quality_flag:
            continue

        decision_time = onset + pd.Timedelta(hours=9)

        with st.container(border=True):

            st.markdown(f"### Storm: {storm_id}")

            s1, s2, s3 = st.columns(3)

            s1.write("**Onset (UTC)**")
            s1.write(format_ist(onset))

            s2.write("**Observed minimum Dst**")
            min_dst = storm.get("min_dst_nt")

            if pd.notna(min_dst):
                s2.write(f"{float(min_dst):.1f} nT")
            else:
                s2.write("Unavailable")

            s3.write("**Detector intensity tier**")
            tier = storm.get("intensity_tier", "Unavailable")
            s3.write(str(tier))

            st.caption(
                "The detector tier is based on the event's "
                "observed data so far; it may change as the "
                "event develops."
            )

            # Require the complete D0-D9 window.
            if decision_time > latest_time:
                st.warning(
                    "WAITING FOR NINE-HOUR WINDOW. "
                    f"Decision time: {format_ist(decision_time)}"
                )

                continue

            eligible_events += 1

            st.markdown("**Nine-Hour AI Prediction**")

            try:

                result = predict_9h_intensity(
                    df,
                    onset
                )

                if not isinstance(result, dict):

                    st.warning(
                        "Unexpected result returned by the "
                        "nine-hour prediction module."
                    )

                    continue

                if result.get("status") == "success":

                    prediction = result.get(
                        "prediction",
                        "Unavailable"
                    )

                    data_quality = result.get(
                        "data_quality",
                        "Not specified"
                    )

                    prediction_text = str(prediction).lower()

                    if "intense" in prediction_text:

                        st.error(
                            f"AI PREDICTION: {prediction}"
                        )

                    elif "moderate" in prediction_text:

                        st.warning(
                            f"AI PREDICTION: {prediction}"
                        )

                    elif "weak" in prediction_text:

                        st.success(
                            f"AI PREDICTION: {prediction}"
                        )

                    else:

                        st.info(
                            f"AI PREDICTION: {prediction}"
                        )

                    st.write(
                        f"**Prediction data quality:** {data_quality}"
                    )

                    if result.get("message"):
                        st.caption(str(result["message"]))

                else:

                    st.warning(
                        result.get(
                            "message",
                            "Prediction is not available."
                        )
                    )

                    if result.get("data_quality"):
                        st.write(result["data_quality"])

            except Exception as e:

                st.error(
                    "Nine-hour prediction failed for this event."
                )

                st.exception(e)

    if eligible_events == 0:

        st.info(
            "No eligible storm event currently has a complete "
            "nine-hour observation window. The dashboard is "
            "waiting for sufficient hourly observations."
        )


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

catalogue = load_catalogue()

st.sidebar.title("🌍 Space Weather AI")

st.sidebar.caption(
    "Geomagnetic Storm Monitoring and Early Warning"
)

st.sidebar.metric(
    "Historical Catalogue Events",
    f"{len(catalogue):,}"
    if not catalogue.empty else "Unavailable"
)

st.sidebar.info(
    "The dashboard uses Kyoto real-time quicklook Dst "
    "for live monitoring."
)

st.sidebar.warning(
    "Quicklook data are preliminary and may contain "
    "noise or baseline shifts. Use provisional or final "
    "Dst data for scientific analysis."
)

# ------------------------------------------------------------
# MAIN TITLE AND TABS
# ------------------------------------------------------------

st.title(
    "🛰️ Geomagnetic Storm Monitor and Early Warning System"
)

st.write(
    "Live Dst monitoring, nine-hour storm-intensity prediction, "
    "historical storm catalogue and model performance."
)

tabs = st.tabs([
    "🌍 Live Monitoring",
    "⚡ 9-Hour Early Warning",
    "📋 Living Storm Catalogue",
    "🧠 1D-CNN Hourly Performance",
    "🔄 Kyoto Auto-Updater"
])


# ============================================================
# TAB 1: LIVE MONITORING
# ============================================================

with tabs[0]:

    st.header("Real-Time Geomagnetic Activity")

    st.info(
        "Source: Kyoto WDC real-time quicklook Dst. "
        "These preliminary readings are intended for monitoring "
        "and forecasting, not as final scientific observations."
    )

    if st.button("🔄 Refresh Live Dst", type="primary"):

        load_live_data.clear()

        st.rerun()

    try:

        raw_df, update_time = load_live_data()

        live_df = prepare_live_data(raw_df)

        if live_df.empty:

            st.warning(
                "No valid live observations are currently available."
            )

        else:

            latest = live_df.iloc[-1]

            latest_dst = float(latest["dst_nt"])

            latest_time = latest["timestamp"]

            age_hours = (
                pd.Timestamp.now(tz="UTC") - latest_time
            ).total_seconds() / 3600

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Latest Dst",
                f"{latest_dst:.1f} nT"
            )

            c2.metric(
                "Latest Observation (IST)",
                format_ist(latest_time)
            )

            c3.metric(
                "Data Age",
                f"{age_hours:.1f} hours"
            )

            if age_hours > 3:

                st.warning(
                    "The latest observation is more than "
                    "three hours old. The feed may be stale."
                )

            else:

                st.success(
                    "Live observations retrieved successfully."
                )

            st.subheader("Current Dst Condition")

            if latest_dst <= -100:

                st.error(
                    "Latest reading is in the intense-level range."
                )

            elif latest_dst <= -50:

                st.warning(
                    "Latest reading is in the moderate-level range."
                )

            elif latest_dst <= -30:

                st.warning(
                    "Latest reading has crossed the conventional "
                    "storm threshold."
                )

            else:

                st.success(
                    "Latest reading is above the conventional "
                    "-30 nT storm threshold."
                )

            st.caption(
                "This is a threshold description, not an AI prediction."
            )

            history_hours = st.selectbox(
                "Select history window",
                [2, 6, 9, 24, 48, 72],
                index=2,
                format_func=lambda x: f"Last {x} hours"
            )

            cutoff = (
                latest_time - pd.Timedelta(hours=history_hours)
            )

            history = live_df[
                live_df["timestamp"] >= cutoff
            ]
            history_ist = history.copy()
            history_ist["timestamp"] = history_ist["timestamp"].dt.tz_convert("Asia/Kolkata")
            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(
                history_ist["timestamp"],
                history_ist["dst_nt"],
                marker="o",
                linewidth=2,
                label="Kyoto Real-Time Dst"
            )

            ax.axhline(
                -30,
                linestyle="--",
                label="-30 nT"
            )

            ax.axhline(
                -50,
                linestyle="--",
                label="-50 nT"
            )

            ax.axhline(
                -100,
                linestyle="--",
                label="-100 nT"
            )

            ax.set_xlabel("Time (IST)")
            ax.set_ylabel("Dst (nT)")
            ax.grid(alpha=0.3)
            ax.legend()

            fig.autofmt_xdate()

            st.pyplot(fig)

            plt.close(fig)

            st.dataframe(
        history_ist.sort_values(
            "timestamp",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

            st.download_button(
    "📥 Download Live Dst CSV",
    live_df.to_csv(index=False).encode("utf-8"),
    "kyoto_live_dst.csv",
    "text/csv",
    key="download_live_dst"
)

    except Exception as e:

        st.error("Unable to retrieve live Kyoto Dst data.")

        st.warning(
            "The source may be temporarily unavailable. "
            "No artificial or simulated readings are displayed."
        )

        st.exception(e)


# ============================================================
# TAB 2: NINE-HOUR EARLY WARNING
# ============================================================

with tabs[1]:

    st.header("Nine-Hour Storm Intensity Prediction")

    st.write(
        "This tab uses the existing live storm detector and "
        "the separate nine-hour prediction module. "
        "The previous six-hour forecaster is not used here."
    )

    st.markdown(
        """
        **Input window:** D0, D1, ..., D9 (10 consecutive hourly readings).

        **Features:** 10 raw Dst readings and 6 engineered features.

        **Intensity classes:**
        - Weak: minimum Dst > -50 nT
        - Moderate: -100 nT < minimum Dst ≤ -50 nT
        - Intense+: minimum Dst ≤ -100 nT
        """
    )

    st.caption(
        "The model predicts an intensity class. It does not predict "
        "an exact future Dst value or exact peak time."
    )

    if st.button(
        "🔄 Refresh and Evaluate 9-Hour Predictions",
        type="primary"
    ):

        load_live_data.clear()

        st.rerun()

    try:

        raw_df, _ = load_live_data()

        live_df = prepare_live_data(raw_df)

        if live_df.empty:

            st.warning(
                "Live observations are unavailable. "
                "Prediction is paused."
            )

        else:

            display_9h_predictions(live_df)

    except Exception as e:

        st.error(
            "Live data could not be retrieved for the nine-hour model."
        )

        st.exception(e)

    st.markdown("---")

    st.subheader("Saved Nine-Hour Model Metrics")

    metrics_path = (
        METRICS_9H_TEST_PATH
        if METRICS_9H_TEST_PATH.exists()
        else METRICS_9H_PATH
    )

    if metrics_path.exists():

        try:

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            st.caption(
                f"Metrics source: {metrics_path.relative_to(ROOT)}"
            )

            metric_keys = [
                "test_accuracy",
                "test_macro_f1",
                "test_weighted_f1",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "validation_accuracy",
                "validation_macro_f1",
                "validation_weighted_f1"
            ]

            available = {
                key: metrics[key]
                for key in metric_keys
                if key in metrics
            }

            if available:
                st.json(available)
            else:
                st.json(metrics)

        except Exception as e:

            st.warning(
                f"Could not read the metrics file: {e}"
            )

    else:

        st.warning(
            "No saved nine-hour metrics JSON was found."
        )


# ============================================================
# TAB 3: LIVING STORM CATALOGUE
# ============================================================

with tabs[2]:

    st.header("Historical Storm Catalogue")

    if catalogue.empty:

        st.warning(
            f"Catalogue unavailable: {CATALOGUE_PATH}"
        )

    else:

        filtered = catalogue.copy()

        if (
            "start_time" in filtered.columns
            and filtered["start_time"].notna().any()
        ):

            years = filtered["start_time"].dt.year.dropna()

            if not years.empty:

                year_range = st.select_slider(
                    "Select year range",
                    options=list(
                        range(
                            int(years.min()),
                            int(years.max()) + 1
                        )
                    ),
                    value=(
                        int(years.min()),
                        int(years.max())
                    )
                )

                filtered = filtered[
                    filtered["start_time"].dt.year.between(
                        year_range[0],
                        year_range[1]
                    )
                ]

        if "intensity_tier" in filtered.columns:

            tiers = sorted(
                filtered["intensity_tier"]
                .dropna()
                .astype(str)
                .unique()
            )

            selected = st.multiselect(
                "Intensity tier",
                tiers,
                default=tiers
            )

            filtered = filtered[
                filtered["intensity_tier"].astype(str).isin(selected)
            ]

        st.metric(
            "Events in current view",
            f"{len(filtered):,}"
        )

        filtered_display = filtered.copy()

        for col in ["start_time", "peak_time", "end_time"]:
            if col in filtered_display.columns:
                filtered_display[col] = (
                    filtered_display[col].dt.tz_convert("Asia/Kolkata")
                )

        st.dataframe(
            filtered_display,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Filtered Catalogue",
            filtered.to_csv(index=False).encode("utf-8"),
            "storm_catalogue_filtered.csv",
            "text/csv",
            key="catalogue_csv_download_button"
        )


# ============================================================
# TAB 4: 1D-CNN PERFORMANCE
# ============================================================

with tabs[3]:

    st.header("1D-CNN Hourly Detection Performance")

    st.write(
        "Historical evaluation metrics from the saved "
        "1D-CNN evaluation file."
    )

    try:

        metrics = load_cnn_metrics()

        if not metrics:

            st.warning(
                f"Metrics file unavailable: {CNN_METRICS_PATH}"
            )

        else:

            hourly = metrics.get("hourly", metrics)

            test = hourly.get(
                "test_at_0.5",
                hourly
            )

            if isinstance(test, dict):

                c1, c2, c3 = st.columns(3)

                for col, label, key in [
                    (c1, "Accuracy", "accuracy"),
                    (c2, "Precision", "precision"),
                    (c3, "F1 Score", "f1")
                ]:

                    value = test.get(key)

                    if isinstance(value, (int, float)):

                        col.metric(
                            label,
                            f"{value * 100:.2f}%"
                        )

                    else:

                        col.metric(label, "Not available")

            with st.expander("View saved metrics JSON"):

                st.json(metrics)

        if FIGURES_DIR.exists():

            figures = sorted(
                list(FIGURES_DIR.glob("*.png"))
                + list(FIGURES_DIR.glob("*.jpg"))
                + list(FIGURES_DIR.glob("*.jpeg"))
            )

            for figure in figures:

                with st.expander(figure.name):

                    st.image(
                        str(figure),
                        use_container_width=True
                    )

    except Exception as e:

        st.error(
            "Unable to load CNN evaluation information."
        )

        st.exception(e)


# ============================================================
# TAB 5: KYOTO AUTO-UPDATER
# ============================================================

with tabs[4]:

    st.header("Kyoto Data and Catalogue Updater")

    st.write(
        "This section provides information about the "
        "Kyoto data source and the project's catalogue updater."
    )

    st.info(
        "The updater is not executed automatically by this dashboard. "
        "The previous updater button was informational only."
    )

    st.markdown("**Kyoto real-time Dst source:**")

    st.link_button(
        "Open Kyoto WDC Real-Time Dst",
        "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/"
    )

    updater_path = ROOT / "src" / "update_catalogue.py"

    st.markdown("**Project catalogue updater command:**")

    st.code(
        "python src/update_catalogue.py",
        language="bash"
    )

    if updater_path.exists():

        st.success(
            "The catalogue updater script exists."
        )

    else:

        st.warning(
            "src/update_catalogue.py was not found."
        )

    st.caption(
        "Run the updater manually from the project root only "
        "when you intend to update the historical catalogue. "
        "Review its output before replacing existing data."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Geomagnetic Storm Monitoring Dashboard | "
    "Kyoto quicklook data are preliminary. "
    "Use provisional or final Dst records for scientific analysis."
)
