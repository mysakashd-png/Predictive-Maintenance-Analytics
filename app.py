
import streamlit as st
import pandas as pd

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Analytics",
    page_icon="⚙️",
    layout="wide"
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚙️ Predictive Maintenance")
st.sidebar.markdown("---")

# ----------------------------
# Main Page
# ----------------------------
st.title("🔧 Predictive Maintenance Analytics Dashboard")

st.markdown("""
Welcome to the **Predictive Maintenance Analytics System**.

This dashboard helps analyze equipment sensor data, detect anomalies,
identify root causes of failures, and build machine learning models
for predictive maintenance.
""")

st.markdown("---")

# ----------------------------
# Dataset Information
# ----------------------------
st.subheader("📊 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Sensors",
        value="9"
    )

with col2:
    st.metric(
        label="Target Variable",
        value="Failure"
    )

with col3:
    st.metric(
        label="Analysis",
        value="Predictive"
    )

st.markdown("---")

# ----------------------------
# Features
# ----------------------------
st.subheader("🚀 Features")

st.markdown("""
✅ Data Analysis

✅ Root Cause Analysis

✅ Anomaly Detection using Z-Score

✅ Rolling Mean & Rolling Standard Deviation

✅ Linear Regression

✅ KNN Regression

✅ Decision Tree Regression

✅ Random Forest Regression

✅ Gradient Boosting Regression

✅ Hyperparameter Tuning

✅ Model Comparison

✅ Failure Prediction
""")

st.markdown("---")

# ----------------------------
# Workflow
# ----------------------------
st.subheader("📌 Workflow")

st.markdown("""
1. Upload Maintenance Dataset

2. Analyze Sensor Data

3. Detect Anomalies

4. Identify Failure Patterns

5. Train Multiple ML Models

6. Compare Model Performance

7. Save Best Model

8. Predict Equipment Failure
""")

st.markdown("---")

# ----------------------------
# Sidebar Instructions
# ----------------------------
st.info(
    "Use the pages in the left sidebar to navigate through the project."
)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Predictive Maintenance Analytics using Streamlit & Machine Learning")
