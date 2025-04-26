import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define L1 Linear Regression class
class L1LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            dw += (self.lambda_param / n_samples) * np.sign(self.weights)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Streamlit Page Setup
st.title('Sleep Quality Prediction App')

# Check if data exists
if 'preprocessed_data' not in st.session_state:
    st.warning('‼️ Preprocessed data not found. Please go to Explore & Preprocess page first.')
    st.stop()

if 'scaler_means' not in st.session_state or 'scaler_stds' not in st.session_state:
    st.warning('‼️ Please go to Train & Evaluate page to train the model with Quality of Sleep first.')
    st.stop()

# Train L1 model if not yet trained
if 'l1_model' not in st.session_state or 'feature_names' not in st.session_state or 'preprocessor' not in st.session_state:
    df = st.session_state['preprocessed_data']

    target_column = 'Quality of Sleep'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse=False), categorical_features)
    ])

    X_transformed = preprocessor.fit_transform(X)

    # Prepare feature names
    numeric_feature_names = numeric_features
    categorical_feature_names = []
    for feature in categorical_features:
        categories = df[feature].dropna().unique()[1:]
        categorical_feature_names.extend([f"{feature}_{cat}" for cat in categories])

    all_feature_names = numeric_feature_names + categorical_feature_names

    l1_model = L1LinearRegression(learning_rate=0.01, n_iterations=1000, lambda_param=0.1)
    l1_model.fit(X_transformed, y)

    # Save to session_state
    st.session_state['l1_model'] = l1_model
    st.session_state['feature_names'] = all_feature_names
    st.session_state['preprocessor'] = preprocessor

# Load model and preprocessor
model = st.session_state['l1_model']
feature_names = st.session_state['feature_names']
preprocessor = st.session_state['preprocessor']

# User Input Form
st.markdown('### Provide your information below to predict your Sleep Quality:')
with st.form("sleep_quality_form"):
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
        stress_level = st.slider("Stress Level", min_value=1, max_value=10, value=6)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=77)
        daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=4200)
        systolic_bp = st.number_input("Systolic BP", min_value=90, max_value=200, value=126)
        diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=120, value=83)
        sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Sleep Apnea", "Insomnia"], index=0)

    submit_button = st.form_submit_button("Predict Sleep Quality")

if submit_button:
    # Build input DataFrame
    user_input = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'BMI Category': [bmi_category],
        'Occupation': [occupation],
        'Sleep Duration': [sleep_duration],
        'Physical Activity Level': [physical_activity],
        'Stress Level': [stress_level],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps],
        'Systolic_BP': [systolic_bp],
        'Diastolic_BP': [diastolic_bp],
        'Sleep Disorder': [sleep_disorder]
    })

    # Feature Engineering
    df = st.session_state['preprocessed_data']
    user_input['Sleep Efficiency'] = df['Sleep Efficiency'].median()
    user_input['Stress_Activity_Ratio'] = stress_level / (physical_activity + 1e-5)

    # Identify numeric and categorical features
    numeric_features = user_input.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = user_input.select_dtypes(include=["object"]).columns.tolist()

    # Encoding
    user_input_encoded = pd.get_dummies(user_input, columns=categorical_features, drop_first=True)

    for feature in feature_names:
        if feature not in user_input_encoded.columns:
            user_input_encoded[feature] = 0

    user_input_final = user_input_encoded[feature_names]

    for col in user_input_final.columns:
        if col in st.session_state['scaler_means']:
            mean = st.session_state['scaler_means'][col]
            std = st.session_state['scaler_stds'][col]
            if std > 0:
                user_input_final[col] = (user_input_final[col] - mean) / std

    # Predict
    predicted_sleep_quality = model.predict(user_input_final.values)[0]

    # Output Result
    st.markdown("### 🌙 Predicted Sleep Quality")
    st.metric("Quality of Sleep", f"{predicted_sleep_quality:.2f}")