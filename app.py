
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "glucose_forecaster.joblib")

EHR_PATH = os.path.join(DATA_DIR, "synthetic_ehr.csv")
PROCESSED_DATA_PATH = os.path.join(
    DATA_DIR, "processed_training_data.csv"
)

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "hba1c",
    "baseline_glucose",
    "glucose",
    "glucose_lag_30m",
    "glucose_lag_60m",
    "glucose_lag_120m",
    "glucose_mean_30m",
    "glucose_mean_60m",
    "heart_rate",
    "steps",
    "sleep_hours",
    "meal_event",
    "meal_carbohydrates_g",
    "hour_sin",
    "hour_cos",
]

TARGET_COLUMN = "target_glucose_2h"
EVENT_THRESHOLD = 180

st.set_page_config(
    page_title="GlucoTwin AI",
    page_icon="🩺",
    layout="wide",
)

# --------------------------------------------------
# 2. Header
# --------------------------------------------------

st.title("🩺 GlucoTwin AI")
st.caption(
    "Synthetic healthcare Digital Twin | "
    "Two-hour glucose forecasting proof of concept"
)

st.info(
    "Research prototype using synthetic data only. "
    "Predictions are not clinically validated and must not be used "
    "for diagnosis, treatment, or medication decisions."
)

# --------------------------------------------------
# 3. Data loading
# --------------------------------------------------

@st.cache_data
def load_data():
    for path in [EHR_PATH, PROCESSED_DATA_PATH]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required project file not found: {path}"
            )

        if os.path.getsize(path) == 0:
            raise ValueError(
                f"Required project file is empty: {path}"
            )

    ehr = pd.read_csv(EHR_PATH)
    data = pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["timestamp"],
    )

    if ehr.empty:
        raise ValueError("The synthetic EHR dataset is empty.")

    if data.empty:
        raise ValueError("The processed sensor dataset is empty.")

    required_ehr = [
        "patient_id",
        "age",
        "sex",
        "bmi",
        "hba1c",
        "diabetes_history",
        "baseline_glucose",
    ]

    required_sensor = [
        "patient_id",
        "timestamp",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    missing_ehr = [
        col for col in required_ehr if col not in ehr.columns
    ]
    missing_sensor = [
        col for col in required_sensor if col not in data.columns
    ]

    if missing_ehr:
        raise ValueError(
            f"Missing EHR columns: {missing_ehr}"
        )

    if missing_sensor:
        raise ValueError(
            f"Missing sensor/model columns: {missing_sensor}"
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data = data.dropna(
        subset=["patient_id", "timestamp", *FEATURE_COLUMNS]
    )
    data = data.sort_values(
        ["patient_id", "timestamp"]
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No valid sensor records remain after data validation."
        )

    return ehr, data


# --------------------------------------------------
# 4. Model loading
# --------------------------------------------------

@st.cache_resource
def load_model():
    """
    Prefer the serialized model if it is compatible with this
    environment. Otherwise train a deployment model using the
    available synthetic processed dataset.

    The fallback model is not the same as the separately evaluated
    held-out model from the original training workflow.
    """

    if os.path.isfile(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)

            if isinstance(bundle, dict):
                model = bundle.get("model")
                features = bundle.get("feature_columns")
                threshold = bundle.get(
                    "event_threshold", EVENT_THRESHOLD
                )

                if (
                    model is not None
                    and features == FEATURE_COLUMNS
                ):
                    return {
                        "model": model,
                        "feature_columns": FEATURE_COLUMNS,
                        "event_threshold": threshold,
                        "model_source": "Serialized model",
                    }

        except Exception:
            # A model serialized in a different library environment
            # may not be compatible. Use the documented fallback.
            pass

    df = pd.read_csv(PROCESSED_DATA_PATH)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Cannot train deployment model; missing columns: {missing}"
        )

    df = df.dropna(subset=required)

    if df.empty:
        raise ValueError(
            "Cannot train deployment model: no valid training rows."
        )

    model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        df[FEATURE_COLUMNS],
        df[TARGET_COLUMN],
    )

    return {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "event_threshold": EVENT_THRESHOLD,
        "model_source": "Deployment model trained on synthetic data",
    }


# --------------------------------------------------
# 5. Initialize the application
# --------------------------------------------------

try:
    ehr, data = load_data()
    bundle = load_model()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = bundle["event_threshold"]
    model_source = bundle["model_source"]

except Exception as exc:
    st.error(f"Could not initialize GlucoTwin AI: {exc}")
    st.stop()


# --------------------------------------------------
# 6. Patient selection
# --------------------------------------------------

st.sidebar.header("Virtual patient")

patient_ids = sorted(
    set(ehr["patient_id"].astype(str))
    & set(data["patient_id"].astype(str))
)

if not patient_ids:
    st.error(
        "No patient IDs are shared between the EHR and sensor datasets."
    )
    st.stop()

patient_id = st.sidebar.selectbox(
    "Select patient",
    patient_ids,
)

patient_data = data[
    data["patient_id"].astype(str) == patient_id
].sort_values("timestamp").reset_index(drop=True)

profile_matches = ehr[
    ehr["patient_id"].astype(str) == patient_id
]

if patient_data.empty or profile_matches.empty:
    st.error("No matching patient profile or sensor records found.")
    st.stop()

profile = profile_matches.iloc[0]

# Offer observations at two-hour intervals to keep the selector
# manageable. Each source observation is still used at its real time.
time_options = patient_data.iloc[::4]["timestamp"].tolist()
last_time = patient_data.iloc[-1]["timestamp"]

if last_time not in time_options:
    time_options.append(last_time)

default_index = min(
    len(time_options) - 1,
    max(0, int(len(time_options) * 0.75)),
)

selected_time = st.sidebar.selectbox(
    "Sensor observation time",
    time_options,
    index=default_index,
    format_func=lambda t: t.strftime("%d %b %Y, %H:%M"),
)

matches = patient_data.index[
    patient_data["timestamp"] == selected_time
].tolist()

if not matches:
    st.error("No matching observation was found.")
    st.stop()

row = patient_data.iloc[matches[0]]


# --------------------------------------------------
# 7. What-if simulation controls
# --------------------------------------------------

st.sidebar.header("What-if simulation")
st.sidebar.caption(
    "Change inputs to explore model sensitivity. "
    "These are simulations, not medical recommendations."
)

meal_carbs = st.sidebar.slider(
    "Current meal carbohydrates (g)",
    min_value=0,
    max_value=120,
    value=int(max(0, min(120, row["meal_carbohydrates_g"]))),
    step=5,
)

steps = st.sidebar.slider(
    "Current interval steps",
    min_value=0,
    max_value=300,
    value=int(max(0, min(300, row["steps"]))),
    step=10,
)

input_row = row[feature_columns].to_frame().T.copy()

# Change only the explicitly selected what-if features.
input_row["meal_carbohydrates_g"] = meal_carbs
input_row["steps"] = steps

prediction = float(
    model.predict(input_row[feature_columns])[0]
)

current_glucose = float(row["glucose"])
actual_future = float(row[TARGET_COLUMN])
forecast_time = selected_time + pd.Timedelta(hours=2)


# --------------------------------------------------
# 8. Key indicators
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Patient ID", patient_id)

c2.metric(
    "Current glucose",
    f"{current_glucose:.1f} mg/dL",
)

c3.metric(
    "Predicted glucose in 2h",
    f"{prediction:.1f} mg/dL",
)

c4.metric(
    "Illustrative threshold flag",
    "Above threshold" if prediction >= threshold
    else "Below threshold",
)

st.caption(
    f"Forecast generated by {model_source}. "
    f"The {threshold} mg/dL threshold is illustrative and is not "
    "a clinical alert recommendation."
)


# --------------------------------------------------
# 9. Patient profile and current sensor readings
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Virtual patient profile")

    st.write(f"**Age:** {int(profile['age'])}")
    st.write(f"**Sex:** {profile['sex']}")
    st.write(f"**BMI:** {profile['bmi']}")
    st.write(f"**HbA1c:** {profile['hba1c']}%")
    st.write(
        f"**Diabetes history:** {profile['diabetes_history']}"
    )
    st.write(
        f"**Baseline glucose:** "
        f"{profile['baseline_glucose']} mg/dL"
    )

with right:
    st.subheader("Current sensor readings")

    st.write(
        f"**Heart rate:** {row['heart_rate']:.1f} bpm"
    )
    st.write(
        f"**Steps in interval:** {int(row['steps'])}"
    )
    st.write(
        f"**Sleep indicator:** {row['sleep_hours']:.3f} hours"
    )
    st.write(
        f"**Meal event:** "
        f"{'Yes' if row['meal_event'] else 'No'}"
    )
    st.write(
        f"**Meal carbohydrates:** "
        f"{int(row['meal_carbohydrates_g'])} g"
    )

st.divider()


# --------------------------------------------------
# 10. Historical glucose and forecast chart
# --------------------------------------------------

st.subheader("Sensor history and two-hour forecast")

# Show the latest 24 hours available before the selected observation.
history = patient_data[
    patient_data["timestamp"] <= selected_time
].tail(48)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=history["timestamp"],
        y=history["glucose"],
        mode="lines+markers",
        name="Observed synthetic glucose",
        line=dict(width=2),
    )
)

# The forecast segment starts at the current observation.
fig.add_trace(
    go.Scatter(
        x=[selected_time, forecast_time],
        y=[current_glucose, prediction],
        mode="lines+markers",
        name="Two-hour forecast",
        line=dict(dash="dash", width=2),
    )
)

# This synthetic target is only available for retrospective comparison.
fig.add_trace(
    go.Scatter(
        x=[forecast_time],
        y=[actual_future],
        mode="markers",
        name="Synthetic future observation (retrospective)",
        marker=dict(symbol="diamond", size=11),
    )
)

fig.add_hline(
    y=threshold,
    line_dash="dot",
    annotation_text="Illustrative threshold",
)

fig.update_layout(
    xaxis_title="Timestamp",
    yaxis_title="Glucose (mg/dL)",
    legend_title="Series",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "The diamond marks the synthetic future outcome recorded in the "
    "dataset for retrospective comparison. It would not be available "
    "at the time of a real-time forecast."
)


# --------------------------------------------------
# 11. What-if comparison
# --------------------------------------------------

st.subheader("What-if comparison")

baseline_input = row[feature_columns].to_frame().T.copy()

baseline_prediction = float(
    model.predict(baseline_input[feature_columns])[0]
)

prediction_change = prediction - baseline_prediction

comparison = pd.DataFrame(
    {
        "Scenario": ["Original inputs", "Modified inputs"],
        "Predicted glucose in 2h (mg/dL)": [
            baseline_prediction,
            prediction,
        ],
        "Meal carbohydrates (g)": [
            int(row["meal_carbohydrates_g"]),
            meal_carbs,
        ],
        "Interval steps": [
            int(row["steps"]),
            steps,
        ],
    }
)

st.dataframe(
    comparison.round(2),
    use_container_width=True,
    hide_index=True,
)

st.metric(
    "Change in predicted glucose",
    f"{prediction_change:+.2f} mg/dL",
    help=(
        "Difference between modified and original model inputs. "
        "This is a model simulation, not a causal or clinical estimate."
    ),
)

st.warning(
    "What-if results are model simulations, not causal estimates or "
    "personalized medical advice. Changes to meal or activity inputs "
    "may not correspond to real physiological effects."
)


# --------------------------------------------------
# 12. Model information
# --------------------------------------------------

st.subheader("Model information")

st.write(f"**Model:** {type(model).__name__}")
st.write(f"**Model source:** {model_source}")
st.write("**Forecast horizon:** 120 minutes")
st.write(f"**Number of model features:** {len(feature_columns)}")
st.write(
    "**Training data:** Artificially generated patient and sensor records"
)

st.caption(
    "The deployment model may differ from the separately evaluated "
    "model. Previously reported held-out test metrics must not be "
    "attributed to this deployment model unless it is independently "
    "evaluated using the same held-out patient split."
)

with st.expander("View current model inputs"):
    st.dataframe(
        input_row[feature_columns],
        use_container_width=True,
    )

st.caption(
    "GlucoTwin AI | Educational proof of concept | "
    "Not for clinical decision-making"
)
