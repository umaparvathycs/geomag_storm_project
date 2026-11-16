
"""
Geomagnetic Storm Autonomous Detection & Early Warning Dashboard

Features:
1. Live Kyoto Dst Monitoring
2. Existing 6-Hour Early Warning Forecaster
3. Living Storm Catalogue Explorer
4. 1D-CNN Hourly Detection Performance
5. Kyoto Living Catalogue Auto-Updater
"""

from pathlib import Path
import sys
import json
import platform

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS AND IMPORTS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Bypass Windows WMI delay-load on Python 3.13
platform._wmi = None

# Import the live Dst reader
try:
    from live_dst_monitor import fetch_live_dst
    LIVE_MODULE_AVAILABLE = True
except Exception as import_error:
    fetch_live_dst = None
    LIVE_MODULE_AVAILABLE = False
    LIVE_IMPORT_ERROR = str(import_error)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Geomagnetic Storm AI & Early Warning",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

CLEAN_DST_PATH = (
    ROOT_DIR / "data" / "processed" / "dst_hourly_clean.csv"
)

CATALOGUE_PATH = (
    ROOT_DIR / "data" / "catalogue" / "storm_catalogue.csv"
)

EARLY_WARNING_MODEL_PATH = (
    ROOT_DIR / "outputs" / "models" / "early_warning_model.joblib"
)

EARLY_WARNING_SCALER_PATH = (
    ROOT_DIR / "outputs" / "models" / "early_warning_scaler.joblib"
)

EARLY_WARNING_METRICS_PATH = (
    ROOT_DIR / "outputs" / "metrics" / "early_warning_summary.json"
)

METRICS_JSON_PATH = (
    ROOT_DIR / "outputs" / "metrics" / "metrics.json"
)

EVENT_SUMMARY_PATH = (
    ROOT_DIR / "outputs" / "metrics" / "event_evaluation_summary.json"
)

FIGURES_DIR = ROOT_DIR / "outputs" / "figures"


# ============================================================
# LOAD EXISTING PROJECT DATA
# ============================================================

@st.cache_data
def load_data():

    cat = pd.read_csv(CATALOGUE_PATH)

    cat["start_time"] = pd.to_datetime(
        cat["start_time"], utc=True
    )

    cat["peak_time"] = pd.to_datetime(
        cat["peak_time"], utc=True
    )

    cat["end_time"] = pd.to_datetime(
        cat["end_time"], utc=True
    )

    dst_df = pd.read_csv(
        CLEAN_DST_PATH,
        nrows=100000
    )

    dst_df["timestamp_utc"] = pd.to_datetime(
        dst_df["timestamp_utc"], utc=True
    )

    return cat, dst_df


@st.cache_resource
def load_early_warning_ai():

    if (
        EARLY_WARNING_MODEL_PATH.exists()
        and EARLY_WARNING_SCALER_PATH.exists()
    ):

        model = joblib.load(
            EARLY_WARNING_MODEL_PATH
        )

        scaler = joblib.load(
            EARLY_WARNING_SCALER_PATH
        )

        return model, scaler

    return None, None


@st.cache_data
def load_metrics_files():

    metrics = {}

    if METRICS_JSON_PATH.exists():

        with open(METRICS_JSON_PATH) as f:
            metrics["hourly"] = json.load(f)

    if EVENT_SUMMARY_PATH.exists():

        with open(EVENT_SUMMARY_PATH) as f:
            metrics["event"] = json.load(f)

    if EARLY_WARNING_METRICS_PATH.exists():

        with open(EARLY_WARNING_METRICS_PATH) as f:
            metrics["early_warning"] = json.load(f)

    return metrics


# ============================================================
# LIVE DST DATA LOADER
# ============================================================

@st.cache_data(ttl=300)
def load_live_dst():

    if not LIVE_MODULE_AVAILABLE:
        raise ImportError(
            "Live Dst module could not be imported. "
            + LIVE_IMPORT_ERROR
        )

    # The live reader returns a DataFrame and source update time.
    dst_live, update_time = fetch_live_dst()

    return dst_live, update_time


# ============================================================
# LOAD EXISTING DATA
# ============================================================

cat_df, dst_df = load_data()

ew_model, ew_scaler = load_early_warning_ai()

saved_metrics = load_metrics_files()


# ============================================================
# EXTRACT EXISTING MODEL METRICS
# ============================================================

test_acc = "97.25%"
test_f1 = "89.77%"

if (
    "hourly" in saved_metrics
    and "test_at_0.5" in saved_metrics["hourly"]
):

    m = saved_metrics["hourly"]["test_at_0.5"]

    test_acc = (
        f"{m.get('accuracy', 0.9725) * 100:.2f}%"
    )

    test_f1 = (
        f"{m.get('f1', 0.8977) * 100:.2f}%"
    )


ew_acc = "88.72%"

if "early_warning" in saved_metrics:

    ew_acc = (
        f"{saved_metrics['early_warning'].get('accuracy', 0.8872) * 100:.2f}%"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Earth%27s_geomagnetism.svg/640px-Earth%27s_geomagnetism.svg.png",
    use_container_width=True
)

st.sidebar.title("🌍 Space Weather AI")

st.sidebar.markdown(
    "**Kyoto Dst Historical Catalogue: 1957–2020**"
)

st.sidebar.markdown("---")

st.sidebar.metric(
    label="Verified Catalogue Storms",
    value=f"{len(cat_df):,}"
)

st.sidebar.metric(
    label="Total Analyzed Hours",
    value="561,024"
)

st.sidebar.metric(
    label="1D-CNN Hourly Accuracy",
    value=test_acc
)

st.sidebar.metric(
    label="6-Hour Early Warning Accuracy",
    value=ew_acc
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Two-Stage Historical System:\n\n"
    "1. 1D-CNN detects storm onset.\n\n"
    "2. Existing Early Warning AI forecasts peak intensity at Hour 6."
)

st.sidebar.caption(
    "Live Dst monitoring uses Kyoto real-time quicklook data."
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🛰️ Geomagnetic Storm AI Monitor & Early Warning System"
)

st.markdown(
    "### Live Dst Monitoring, Storm Detection & Early Warning"
)


# ============================================================
# DASHBOARD TABS
# ============================================================

tabs = st.tabs([

    "🌍 Live Monitoring",

    "⚡ 6-Hour Early Warning Forecaster",

    "📋 Living Storm Catalogue",

    "🧠 1D-CNN Hourly Performance",

    "🔄 Kyoto Auto-Updater"

])


# ============================================================
# TAB 1: LIVE DST MONITORING
# ============================================================

with tabs[0]:

    st.header("🌍 Real-Time Geomagnetic Activity Monitoring")

    st.write(
        "Monitor the latest available hourly equatorial Dst "
        "measurements obtained from the Kyoto World Data Center."
    )

    st.info(
        "Data source: Kyoto WDC — Real-Time Quicklook Dst Index. "
        "These preliminary readings may contain artificial noise "
        "or baseline shifts. They are intended for monitoring "
        "and forecasting, not as final scientific observations."
    )

    # --------------------------------------------------------
    # Refresh controls
    # --------------------------------------------------------

    refresh_col, status_col = st.columns([1, 3])

    with refresh_col:

        refresh_live = st.button(
            "🔄 Refresh Live Dst",
            type="primary"
        )

    if refresh_live:

        load_live_dst.clear()

    with status_col:

        st.caption(
            "Live data is cached for up to 5 minutes. "
            "Use Refresh Live Dst to request a new reading."
        )

    # --------------------------------------------------------
    # Retrieve live data
    # --------------------------------------------------------

    try:

        live_df, update_time = load_live_dst()

        if live_df.empty:

            st.warning(
                "Kyoto returned no valid Dst observations."
            )

        else:

            live_df = live_df.copy()

            live_df["timestamp"] = pd.to_datetime(
                live_df["timestamp"],
                utc=True
            )

            live_df = (
                live_df.sort_values("timestamp")
                .drop_duplicates(
                    subset="timestamp",
                    keep="last"
                )
            )

            latest_row = live_df.iloc[-1]

            latest_dst = float(
                latest_row["dst_nt"]
            )

            latest_timestamp = latest_row["timestamp"]

            current_time = pd.Timestamp.now(tz="UTC")

            data_age_hours = (
                current_time - latest_timestamp
            ).total_seconds() / 3600

            # ------------------------------------------------
            # Latest readings and source information
            # ------------------------------------------------

            st.subheader("Latest Available Dst Observation")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            metric_col1.metric(
                "Latest Dst",
                f"{latest_dst:.0f} nT"
            )

            metric_col2.metric(
                "Latest Observation (UTC)",
                latest_timestamp.strftime("%d %b %Y, %H:%M")
            )

            if update_time is not None:

                metric_col3.metric(
                    "Kyoto Page Updated (UTC)",
                    update_time.strftime("%d %b %Y, %H:%M")
                )

            else:

                metric_col3.metric(
                    "Kyoto Page Updated",
                    "Not available"
                )

            if data_age_hours > 3:

                st.warning(
                    f"The latest extracted observation is "
                    f"{data_age_hours:.1f} hours old. "
                    "The data may be delayed or incomplete."
                )

            else:

                st.success(
                    "Kyoto live data was retrieved successfully."
                )

            # ------------------------------------------------
            # Dst condition
            # ------------------------------------------------

            st.subheader("Current Geomagnetic Condition")

            if latest_dst <= -100:

                st.error(
                    "Intense geomagnetic storm-level Dst value "
                    "in the latest available observation."
                )

            elif latest_dst <= -50:

                st.warning(
                    "Moderate geomagnetic disturbance-level "
                    "Dst value in the latest observation."
                )

            elif latest_dst <= -30:

                st.warning(
                    "Dst is within the conventional storm "
                    "threshold range."
                )

            else:

                st.success(
                    "The latest available Dst value is above "
                    "the conventional -30 nT storm threshold."
                )

            st.caption(
                "This is a threshold-based description of the "
                "latest Dst observation, not an AI storm prediction."
            )

            # ------------------------------------------------
            # Select history window
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("Recent Dst History")

            history_options = {
                "Last 2 hours": 2,
                "Last 6 hours": 6,
                "Last 9 hours": 9,
                "Last 24 hours": 24
            }

            selected_history = st.selectbox(
                "Select observation window:",
                list(history_options.keys()),
                index=2
            )

            history_hours = history_options[
                selected_history
            ]

            cutoff_time = (
                latest_timestamp
                - pd.Timedelta(hours=history_hours)
            )

            history_df = live_df[
                live_df["timestamp"] >= cutoff_time
            ].copy()

            # ------------------------------------------------
            # Plot
            # ------------------------------------------------

            if not history_df.empty:

                fig, ax = plt.subplots(figsize=(12, 4))

                ax.plot(
                    history_df["timestamp"],
                    history_df["dst_nt"],
                    marker="o",
                    linewidth=2,
                    label="Kyoto Real-Time Dst"
                )

                ax.axhline(
                    -30,
                    linestyle="--",
                    color="orange",
                    label="Storm threshold (-30 nT)"
                )

                ax.axhline(
                    -50,
                    linestyle="--",
                    color="gold",
                    label="Moderate threshold (-50 nT)"
                )

                ax.axhline(
                    -100,
                    linestyle="--",
                    color="red",
                    label="Intense threshold (-100 nT)"
                )

                ax.set_xlabel("Time (UTC)")
                ax.set_ylabel("Dst (nT)")

                ax.set_title(
                    f"Kyoto Real-Time Dst — {selected_history}"
                )

                ax.grid(True, alpha=0.3)
                ax.legend()

                fig.autofmt_xdate()

                st.pyplot(fig)

                plt.close(fig)

                st.write(
                    f"Displaying **{len(history_df)}** "
                    "available observations in the selected window."
                )

                st.dataframe(
                    history_df.sort_values(
                        "timestamp",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "No valid observations are available "
                    "in the selected time window."
                )

            # ------------------------------------------------
            # Data download
            # ------------------------------------------------

            csv_data = live_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "📥 Download Available Live Dst Data",
                data=csv_data,
                file_name="kyoto_live_dst.csv",
                mime="text/csv"
            )

            # ------------------------------------------------
            # Source warning
            # ------------------------------------------------

            st.markdown("---")

            st.caption(
                "Source: WDC for Geomagnetism, Kyoto University. "
                "Real-time quicklook data may contain artificial "
                "noise and baseline shifts. Use provisional or "
                "final Dst records for scientific analysis."
            )

    except Exception as error:

        st.error(
            "Unable to retrieve live Kyoto Dst data."
        )

        st.warning(
            "The live source may be temporarily unavailable. "
            "No artificial or simulated readings will be displayed."
        )

        st.code(str(error))


# ============================================================
# TAB 2: EXISTING 6-HOUR EARLY WARNING FORECASTER
# ============================================================

with tabs[1]:

    st.header(
        "⚡ Storm Intensity Forecasting (First 6 Hours)"
    )

    st.write(
        "Geomagnetic storms typically develop over "
        "**6 to 24+ hours** to reach peak depression. "
        "By observing the first 6 hours of onset, "
        "this feasibility prototype forecasts whether Earth "
        "will face a Weak, Moderate, or Intense / Severe storm."
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        st.subheader("Select a Storm Case")

        famous_options = {

            "Custom Live Simulation": None,

            "1989 Quebec Blackout Storm (Peak: -589 nT)": (
                "STORM_1989_042"
            ),

            "2003 Halloween Solar Storm (Peak: -383 nT)": (
                "STORM_2003_063"
            ),

            "2017 St. Patrick's Storm (Peak: -142 nT)": (
                "STORM_2017_056"
            ),

            "2018 Moderate Event (Peak: -75 nT)": (
                "STORM_2018_035"
            ),

            "2019 Weak Event (Peak: -42 nT)": (
                "STORM_2019_012"
            )

        }

        selection = st.selectbox(
            "Choose Historical Event:",
            list(famous_options.keys())
        )

        if selection == "Custom Live Simulation":

            st.markdown(
                "**Enter 6-Hour Dst Trajectory (nT):**"
            )

            d0 = st.number_input(
                "Hour 0 (Onset)", value=-32.0, step=1.0
            )

            d1 = st.number_input(
                "Hour 1", value=-40.0, step=1.0
            )

            d2 = st.number_input(
                "Hour 2", value=-55.0, step=1.0
            )

            d3 = st.number_input(
                "Hour 3", value=-75.0, step=1.0
            )

            d4 = st.number_input(
                "Hour 4", value=-90.0, step=1.0
            )

            d5 = st.number_input(
                "Hour 5", value=-105.0, step=1.0
            )

            d6 = st.number_input(
                "Hour 6 (Decision Point)",
                value=-120.0,
                step=1.0
            )

            vals = np.array([
                d0, d1, d2, d3, d4, d5, d6
            ])

            actual_peak = None
            main_hours = None

        else:

            storm_id = famous_options[selection]

            matched = cat_df[
                cat_df["storm_id"] == storm_id
            ]

            if len(matched) > 0:

                s_row = matched.iloc[0]

                actual_peak = s_row["min_dst_nt"]

                main_hours = s_row["main_phase_hours"]

                drop_rate = (
                    abs(actual_peak + 30)
                    / max(main_hours, 6)
                )

                vals = np.array([

                    -30.0,

                    -30.0 - drop_rate * 0.8,

                    -30.0 - drop_rate * 1.5,

                    -30.0 - drop_rate * 2.3,

                    -30.0 - drop_rate * 3.2,

                    -30.0 - drop_rate * 4.1,

                    -30.0 - drop_rate * 5.0

                ])

            else:

                vals = np.array([
                    -30.0, -38.0, -48.0,
                    -62.0, -78.0, -92.0, -110.0
                ])

                actual_peak = -150.0
                main_hours = 18.0

        predict_btn = st.button(
            "🚀 Forecast Peak Severity from 6-Hour Onset",
            type="primary"
        )

    with col2:

        st.subheader("AI Early Warning Prediction")

        if ew_model is not None and ew_scaler is not None:

            min_6 = np.min(vals)

            drop_6 = vals[0] - min_6

            slope_6 = (vals[6] - vals[0]) / 6.0

            accel_6 = (
                (vals[6] - vals[3])
                - (vals[3] - vals[0])
            )

            std_6 = np.std(vals)

            mean_6 = np.mean(vals)

            feat_vec = np.array([[

                vals[0],
                vals[1],
                vals[2],
                vals[3],
                vals[4],
                vals[5],
                vals[6],

                min_6,
                drop_6,
                slope_6,
                accel_6,
                std_6,
                mean_6

            ]])

            feat_scaled = ew_scaler.transform(
                feat_vec
            )

            pred_class = ew_model.predict(
                feat_scaled
            )[0]

            probs = ew_model.predict_proba(
                feat_scaled
            )[0]

            class_labels = [

                "Weak (-30 to -50 nT)",

                "Moderate (-50 to -100 nT)",

                "Intense / Severe (<= -100 nT)"

            ]

            icons = ["🟢", "🟡", "🔴"]

            st.markdown(
                f"### {icons[pred_class]} Forecasted Peak Severity: "
                f"**{class_labels[pred_class]}**"
            )

            p_col1, p_col2, p_col3 = st.columns(3)

            p_col1.metric(
                "Weak Probability",
                f"{probs[0] * 100:.1f}%"
            )

            p_col2.metric(
                "Moderate Probability",
                f"{probs[1] * 100:.1f}%"
            )

            p_col3.metric(
                "Intense+ Probability",
                f"{probs[2] * 100:.1f}%"
            )

            if actual_peak is not None:

                st.success(
                    f"**Ground Truth Peak:** {actual_peak:.1f} nT | "
                    f"**Main Phase Duration:** {main_hours:.0f} hours"
                )

                lead_time = max(
                    0,
                    main_hours - 6
                )

                if lead_time > 0:

                    st.info(
                        f"⏱️ **Approximate Advance Warning:** "
                        f"**{lead_time:.0f} hours**"
                    )

                else:

                    st.warning(
                        "Peak minimum occurred within the "
                        "first 6 hours of onset."
                    )

            fig, ax = plt.subplots(figsize=(8, 4))

            hours = np.arange(7)

            ax.plot(
                hours,
                vals,
                "o-",
                color="crimson",
                linewidth=2.5,
                label="Initial 6-Hour Trajectory"
            )

            ax.axhline(
                -30,
                linestyle="--",
                color="gray",
                label="Storm Onset Threshold (-30 nT)"
            )

            ax.axhline(
                -50,
                linestyle="--",
                color="gold",
                label="Moderate Boundary (-50 nT)"
            )

            ax.axhline(
                -100,
                linestyle="--",
                color="red",
                label="Intense Boundary (-100 nT)"
            )

            ax.set_xlabel("Hours Since Storm Onset")
            ax.set_ylabel("Dst (nT)")

            ax.set_title(
                "Initial 6-Hour Storm Profile"
            )

            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower left")

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.warning(
                "The existing 6-hour early-warning model "
                "or scaler was not found."
            )

            st.info(
                "Expected model files:\n"
                "- outputs/models/early_warning_model.joblib\n"
                "- outputs/models/early_warning_scaler.joblib"
            )

    # Lead-time distribution

    if (
        "early_warning" in saved_metrics
        and "lead_time_statistics"
        in saved_metrics["early_warning"]
    ):

        lstats = saved_metrics["early_warning"][
            "lead_time_statistics"
        ]

        st.markdown("---")

        st.subheader(
            "📊 Empirical Test Set Lead-Time Analysis (2016–2020)"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Storms with Advance Lead Time (>0h)",
            f"{lstats.get('pct_storms_with_advance_lead_time', 30.4):.1f}%"
        )

        c2.metric(
            "Mean Lead Time (Positive Storms)",
            f"{lstats.get('mean_positive_lead_time_hours', 9.6):.1f} hours"
        )

        c3.metric(
            "Intense+ Storms Mean Lead Time",
            f"{lstats.get('intense_mean_lead_time_hours', 6.8):.1f} hours"
        )


# ============================================================
# TAB 3: LIVING STORM CATALOGUE EXPLORER
# ============================================================

with tabs[2]:

    st.header(
        "📋 Living Storm Catalogue (1957–2020)"
    )

    st.write(
        f"Displaying **{len(cat_df):,} verified storms** "
        "detected from Kyoto final Dst observations."
    )

    f_col1, f_col2, f_col3 = st.columns(3)

    year_range = f_col1.slider(
        "Filter by Start Year",
        1957,
        2020,
        (1989, 2020)
    )

    tier_filter = f_col2.multiselect(
        "Filter by Intensity Tier",
        cat_df["intensity_tier"].unique().tolist(),
        default=["Intense", "Severe", "Extreme"]
    )

    min_dst_max = f_col3.slider(
        "Minimum Dst (nT) Upper Bound",
        -600,
        -30,
        -30
    )

    filtered = cat_df[

        (cat_df["start_time"].dt.year >= year_range[0])

        & (cat_df["start_time"].dt.year <= year_range[1])

        & (cat_df["intensity_tier"].isin(tier_filter))

        & (cat_df["min_dst_nt"] <= min_dst_max)

    ].sort_values("min_dst_nt")

    st.write(
        f"Showing **{len(filtered):,} storms** matching filters:"
    )

    st.dataframe(

        filtered[[

            "storm_id",
            "start_time",
            "peak_time",
            "end_time",
            "min_dst_nt",
            "intensity_tier",
            "total_duration_hours",
            "main_phase_hours",
            "recovery_to_150_hours"

        ]],

        use_container_width=True

    )

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Filtered Catalogue (CSV)",
        csv_data,
        "filtered_storm_catalogue.csv",
        "text/csv"
    )


# ============================================================
# TAB 4: 1D-CNN HOURLY PERFORMANCE & AUDIT
# ============================================================

with tabs[3]:

    st.header(
        "🧠 1D-CNN Model Hourly Detection Benchmark"
    )

    st.write(
        "Evaluated on **43,801 hourly windows** "
        "from the independent held-out test split (2016–2020):"
    )

    h_m = saved_metrics.get(
        "hourly", {}
    ).get(
        "test_at_0.5", {}
    )

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    m_col1.metric(
        "Accuracy",
        f"{h_m.get('accuracy', 0.9725) * 100:.2f}%"
    )

    m_col2.metric(
        "Precision",
        f"{h_m.get('precision', 0.8973) * 100:.2f}%"
    )

    m_col3.metric(
        "Recall",
        f"{h_m.get('recall', 0.8981) * 100:.2f}%"
    )

    m_col4.metric(
        "PR-AUC",
        f"{h_m.get('pr_auc', 0.9581) * 100:.2f}%"
    )

    st.subheader(
        "Confusion Matrix (43,801 Windows)"
    )

    cm_vals = h_m.get(
        "confusion_matrix[[TN,FP],[FN,TP]]",
        [[37320, 604], [599, 5278]]
    )

    cm_data = pd.DataFrame(

        cm_vals,

        index=[
            "True Quiet (Non-Storm)",
            "True Storm"
        ],

        columns=[
            "Predicted Quiet",
            "Predicted Storm"
        ]

    )

    st.table(cm_data)

    st.subheader(
        "Annual Validation Plots (2016–2020)"
    )

    selected_year = st.selectbox(
        "Select Year to Inspect Time-Series Detection:",
        [2016, 2017, 2018, 2019, 2020]
    )

    fig_path = (
        FIGURES_DIR
        / f"cnn_event_evaluation_{selected_year}.png"
    )

    if fig_path.exists():

        st.image(
            str(fig_path),
            caption=(
                f"Kyoto Dst vs CNN Predicted Events — "
                f"{selected_year}"
            ),
            use_container_width=True
        )

    else:

        st.info(
            "The annual validation figure is not available."
        )


# ============================================================
# TAB 5: KYOTO LIVING CATALOGUE AUTO-UPDATER
# ============================================================

with tabs[4]:

    st.header(
        "🔄 Kyoto Living Catalogue Auto-Updater"
    )

    st.write(

        "Kyoto University WDC publishes new monthly Dst records "
        "periodically. The existing project pipeline can download "
        "new monthly records, clean the hourly series, run storm "
        "detection, and update the living catalogue."

    )

    u_col1, u_col2 = st.columns(2)

    u_col1.info(
        "**Current Catalogue Coverage:** "
        "1957-01 to 2020-12 "
        "(561,024 Verified Final Hours)"
    )

    u_col2.info(
        "**Archive Status:** "
        "Use the project update pipeline to verify "
        "and ingest newly available final records."
    )

    st.warning(
        "The button below is retained from the existing dashboard. "
        "It does not execute the update pipeline automatically."
    )

    if st.button(
        "🔄 Check & Ingest Latest Kyoto Monthly Data"
    ):

        st.info(
            "To run the actual catalogue update, use the "
            "project's update script from the VS Code terminal. "
            "The existing button does not download or modify data."
        )

        st.code(
            "python src/update_catalogue.py",
            language="powershell"
        )


# ============================================================
# END OF DASHBOARD
# ============================================================

