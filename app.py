import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "glucose_forecaster.joblib")

st.set_page_config(
    page_title="GlucoTwin AI",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 GlucoTwin AI")
st.caption(
    "Synthetic healthcare Digital Twin | "
    "Two-hour glucose forecasting proof of concept"
)

st.info(
    "Research prototype using synthetic data only. "
    "Predictions are not clinically validated and must not be used "
    "for diagnosis, treatment or medication decisions."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    ehr = pd.read_csv(os.path.join(DATA_DIR, "synthetic_ehr.csv"))
    data = pd.read_csv(
        os.path.join(DATA_DIR, "processed_training_data.csv"),
        parse_dates=["timestamp"]
    )
    return ehr, data

try:
    bundle = load_model()
    ehr, data = load_data()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = bundle["event_threshold"]
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.stop()

st.sidebar.header("Virtual patient")

patient_ids = sorted(ehr["patient_id"].unique())
patient_id = st.sidebar.selectbox("Select patient", patient_ids)

patient_data = data[data["patient_id"] == patient_id].sort_values(
    "timestamp"
).reset_index(drop=True)

# Show one selectable timestamp per hour to keep the UI responsive.
time_options = patient_data.iloc[::12]["timestamp"].tolist()
if patient_data.iloc[-1]["timestamp"] not in time_options:
    time_options.append(patient_data.iloc[-1]["timestamp"])

selected_time = st.sidebar.selectbox(
    "Sensor observation time",
    time_options,
    format_func=lambda t: t.strftime("%d %b %Y, %H:%M")
)

matches = patient_data.index[
    patient_data["timestamp"] == selected_time
].tolist()

if not matches:
    st.error("No matching observation was found.")
    st.stop()

row_index = matches[0]
row = patient_data.iloc[row_index]
profile = ehr[ehr["patient_id"] == patient_id].iloc[0]

# Illustrative what-if controls.
st.sidebar.header("What-if simulation")
st.sidebar.caption("Change inputs to explore model sensitivity.")

meal_carbs = st.sidebar.slider(
    "Current meal carbohydrates (g)", 0, 120,
    int(row["meal_carbohydrates_g"]), 5
)

steps = st.sidebar.slider(
    "Current interval steps", 0, 300,
    int(min(row["steps"], 300)), 10
)

input_row = row[feature_columns].to_frame().T.copy()
input_row["meal_carbohydrates_g"] = meal_carbs
input_row["steps"] = steps

# The saved trained model generates the actual forecast.
prediction = float(model.predict(input_row[feature_columns])[0])

# These metrics describe the selected simulated observation.
current_glucose = float(row["glucose"])
actual_future = float(row["target_glucose_2h"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patient ID", patient_id)
c2.metric("Current glucose", f"{current_glucose:.1f} mg/dL")
c3.metric("Predicted glucose in 2h", f"{prediction:.1f} mg/dL")
c4.metric(
    "Illustrative threshold flag",
    "Above threshold" if prediction >= threshold else "Below threshold"
)

st.caption(
    f"Forecast generated from the saved model. The {threshold} mg/dL "
    "threshold is illustrative, not a clinical alert recommendation."
)

left, right = st.columns(2)

with left:
    st.subheader("Virtual patient profile")
    st.write(f"**Age:** {int(profile['age'])}")
    st.write(f"**Sex:** {profile['sex']}")
    st.write(f"**BMI:** {profile['bmi']}")
    st.write(f"**HbA1c:** {profile['hba1c']}%")
    st.write(f"**Diabetes history:** {profile['diabetes_history']}")
    st.write(f"**Baseline glucose:** {profile['baseline_glucose']} mg/dL")

with right:
    st.subheader("Current sensor readings")
    st.write(f"**Heart rate:** {row['heart_rate']:.1f} bpm")
    st.write(f"**Steps in interval:** {int(row['steps'])}")
    st.write(f"**Sleep indicator:** {row['sleep_hours']:.3f} hours")
    st.write(f"**Meal event:** {'Yes' if row['meal_event'] else 'No'}")
    st.write(f"**Meal carbohydrates:** {int(row['meal_carbohydrates_g'])} g")

st.divider()
st.subheader("Sensor history and two-hour forecast")

history = patient_data[
    (patient_data["timestamp"] <= selected_time)
].tail(72)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=history["timestamp"],
    y=history["glucose"],
    mode="lines",
    name="Observed synthetic glucose"
))

forecast_time = selected_time + pd.Timedelta(hours=2)

fig.add_trace(go.Scatter(
    x=[selected_time, forecast_time],
    y=[current_glucose, prediction],
    mode="lines+markers",
    name="Forecast trajectory",
    line=dict(dash="dash")
))

# The outcome exists only because this is a retrospective synthetic dataset.
fig.add_trace(go.Scatter(
    x=[forecast_time],
    y=[actual_future],
    mode="markers",
    name="Simulated future observation (retrospective)",
    marker=dict(symbol="diamond", size=11)
))

fig.add_hline(
    y=threshold,
    line_dash="dot",
    annotation_text="Illustrative threshold"
)

fig.update_layout(
    xaxis_title="Timestamp",
    yaxis_title="Glucose (mg/dL)",
    legend_title="Series",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "The diamond shows the synthetic outcome already present in the "
    "dataset for retrospective evaluation. It would not be available "
    "when making a real-time forecast."
)

st.subheader("What-if comparison")

baseline_input = row[feature_columns].to_frame().T
baseline_prediction = float(
    model.predict(baseline_input[feature_columns])[0]
)

comparison = pd.DataFrame({
    "Scenario": ["Original inputs", "Modified inputs"],
    "Predicted glucose in 2h (mg/dL)": [
        baseline_prediction, prediction
    ],
    "Meal carbohydrates (g)": [
        int(row["meal_carbohydrates_g"]), meal_carbs
    ],
    "Interval steps": [int(row["steps"]), steps]
})

st.dataframe(comparison, use_container_width=True, hide_index=True)

st.warning(
    "What-if results are model simulations, not causal estimates or "
    "personalized medical advice. Changes to meal or activity inputs "
    "may not correspond to real physiological effects."
)

st.subheader("Model information")
st.write("**Model:** HistGradientBoostingRegressor")
st.write("**Forecast horizon:** 120 minutes")
st.write(f"**Number of model features:** {len(feature_columns)}")
st.write("**Data:** Artificially generated patient and sensor records")

with st.expander("View current model inputs"):
    st.dataframe(input_row, use_container_width=True)

st.caption("GlucoTwin AI | Educational proof of concept")
