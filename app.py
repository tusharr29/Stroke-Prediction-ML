
import streamlit as st
import pandas as pd
import joblib

# Load trained model
artifact = joblib.load("stroke_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]

# Page configuration
st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🧠",
    layout="centered"
)

# Application heading
st.title("Stroke Prediction System")

st.write(
    "Enter the patient details below to generate "
    "a prediction using our Logistic Regression model."
)

st.warning(
    "Educational demonstration only. "
    "This tool is not a medical diagnostic system."
)

# Patient information form
st.header("Patient Information")

with st.form("patient_form"):

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=120.0,
        value=45.0
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    ever_married = st.selectbox(
        "Ever Married",
        ["Yes", "No"]
    )

    work_type = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked"
        ]
    )

    Residence_type = st.selectbox(
        "Residence Type",
        ["Urban", "Rural"]
    )

    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=0.0,
        value=100.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        [
            "formerly smoked",
            "never smoked",
            "smokes",
            "Unknown"
        ]
    )

    submit = st.form_submit_button("Predict")

# Prediction
if submit:

    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= threshold)

    st.subheader("Prediction Result")

    st.metric(
        "Model-estimated stroke probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("The model classified this input as Class 1.")
    else:
        st.success("The model classified this input as Class 0.")

    st.caption(
        f"Classification threshold: {threshold:.2f}"
    )
