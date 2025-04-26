import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

st.title('Deploy Application')

# Check if trained model and required information exist
model = st.session_state.get('l2_model', None)
feature_names = st.session_state.get('feature_names', None)

# Check if model and data are ready
if model is None or feature_names is None:
    st.warning('‼️ The Sleep Health Application is under construction. Please train the model first.')
    st.stop()

# Check if trained with correct target
if st.session_state['target_column'] != 'Stress Level':
    st.warning("‼️ The model was not trained with 'Stress Level' as target. Please retrain.")
    st.stop()

# Deploy App
st.markdown('### Sleep Stress Level Prediction')

st.markdown('#### Provide your information below to predict your stress level:')

# User input form
with st.form("stress_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=27)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        bmi_category = st.selectbox("BMI Category", ["Normal Weight", "Overweight", "Obese"], index=1)
        occupation = st.selectbox("Occupation", [
            "Software Engineer", "Doctor", "Sales Representative", "Teacher", 
            "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", 
            "Salesperson", "Manager"
        ], index=0)
        sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=6.1, step=0.1)
        physical_activity = st.number_input("Physical Activity Level (%)", min_value=0.0, max_value=100.0, value=42.0)

    with col2:
        quality_of_sleep = st.slider("Quality of Sleep", min_value=1, max_value=10, value=6)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=77)
        daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=4200)
        systolic_bp = st.number_input("Systolic BP", min_value=90, max_value=200, value=126)
        diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=120, value=83)
        sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Sleep Apnea", "Insomnia"], index=0)

    submit_button = st.form_submit_button("Predict Stress Level")

if submit_button:
    # Build input DataFrame
    user_input = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'BMI Category': [bmi_category],
        'Occupation': [occupation],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps],
        'Systolic_BP': [systolic_bp],
        'Diastolic_BP': [diastolic_bp],
        'Sleep Disorder': [sleep_disorder]
    })

    # Feature Engineering
    user_input['Sleep Efficiency'] = quality_of_sleep / (sleep_duration + 1e-5)  # Assume 8 hours as baseline for Sleep Efficiency
    df = st.session_state['preprocessed_data']
    user_input['Stress_Activity_Ratio'] = 0

    # Identify numeric and categorical features
    numeric_features = user_input.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = user_input.select_dtypes(include=["object"]).columns.tolist()

    # Build Preprocessor
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit and transform (demo purpose)
    user_input_processed = preprocessor.fit_transform(user_input)

    # Create processed feature names
    processed_feature_names = (
        preprocessor.named_transformers_["num"].get_feature_names_out(numeric_features).tolist()
        + preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features).tolist()
    )

    # Create DataFrame
    user_input_final = pd.DataFrame(user_input_processed, columns=processed_feature_names)

    # Ensure all expected features exist
    for feature in feature_names:
        if feature not in user_input_final.columns:
            user_input_final[feature] = 0

    # Reorder columns
    user_input_final = user_input_final[feature_names]

    # Predict
    predicted_stress_level = model.predict(user_input_final.values)[0]
    predicted_stress_level *= 10

    # Output result
    st.markdown("### 🧠 Predicted Stress Level")
    st.metric("Stress Level (0-10 Scale)", f"{predicted_stress_level:.2f}")

    st.markdown("#### Personalized Suggestion")
    if predicted_stress_level < 4:
        st.success("Low Stress: Keep up the good lifestyle habits!")
    elif 4 <= predicted_stress_level < 7:
        st.info("Moderate Stress: Consider better relaxation and sleep practices.")
    else:
        st.warning("High Stress: Please prioritize your mental and physical well-being.")
