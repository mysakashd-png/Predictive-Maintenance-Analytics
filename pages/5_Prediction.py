
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Failure Prediction",
    layout="wide"
)

st.title("🔮 Failure Prediction")

# ------------------------------------
# Load Model
# ------------------------------------

try:
    model = joblib.load(
        "models/best_model.pkl"
    )

except:
    st.error(
        "Model not found. Please train and save a model first."
    )
    st.stop()

# ------------------------------------
# User Inputs
# ------------------------------------

st.subheader(
    "Enter Sensor Values"
)

col1, col2, col3 = st.columns(3)

with col1:
    metric1 = st.number_input(
        "Metric 1",
        value=0.0
    )

    metric2 = st.number_input(
        "Metric 2",
        value=0.0
    )

    metric3 = st.number_input(
        "Metric 3",
        value=0.0
    )

with col2:
    metric4 = st.number_input(
        "Metric 4",
        value=0.0
    )

    metric5 = st.number_input(
        "Metric 5",
        value=0.0
    )

    metric6 = st.number_input(
        "Metric 6",
        value=0.0
    )

with col3:
    metric7 = st.number_input(
        "Metric 7",
        value=0.0
    )

    metric8 = st.number_input(
        "Metric 8",
        value=0.0
    )

    metric9 = st.number_input(
        "Metric 9",
        value=0.0
    )

# ------------------------------------
# Prediction
# ------------------------------------

if st.button(
    "Predict Failure"
):

    input_data = pd.DataFrame({
        "metric1":[metric1],
        "metric2":[metric2],
        "metric3":[metric3],
        "metric4":[metric4],
        "metric5":[metric5],
        "metric6":[metric6],
        "metric7":[metric7],
        "metric8":[metric8],
        "metric9":[metric9]
    })

    prediction = model.predict(
        input_data
    )[0]

    st.subheader(
        "Prediction Result"
    )

    st.metric(
        "Predicted Value",
        round(float(prediction), 4)
    )

    # --------------------------------
    # Failure Status
    # --------------------------------

    if prediction >= 0.5:

        st.error(
            "⚠️ High Failure Risk Detected"
        )

    else:

        st.success(
            "✅ Equipment Operating Normally"
        )

    # --------------------------------
    # Input Summary
    # --------------------------------

    st.subheader(
        "Input Summary"
    )

    st.dataframe(
        input_data,
        use_container_width=True
    )
