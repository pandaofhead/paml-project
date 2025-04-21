import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from helper_functions import fetch_dataset

# Page Setup
st.set_page_config(page_title="Sleep Health and Lifestyle Dataset", layout="wide")
st.title("Train Machine Learning Models")

# Check for preprocessed data
if 'preprocessed_data' in st.session_state:
    df = st.session_state['preprocessed_data']
    st.success("Using preprocessed dataset from Explore/Preprocess page.")
else:
    st.warning("‼️ Preprocessed data not found. Please go to **Explore & Preprocess** and complete preprocessing first.")
    st.stop()

# Main UI
st.markdown("### 📂 Dataset Preview")
st.dataframe(df.head())

st.markdown("### 🎯 Select Target and Features")
target_column = st.selectbox("Target Variable", df.columns, index=df.columns.get_loc("Quality of Sleep") if "Quality of Sleep" in df.columns else 0)

input_columns = df.columns[df.columns != target_column].tolist()
selected_features = st.multiselect("Input Features", input_columns, default=input_columns)

X = df[selected_features]
y = df[target_column]

# Split Data
st.markdown("### 🔀 Split Dataset")
test_size = st.slider("Test Size (%)", 10, 50, 30)
val_size = st.slider("Validation Size (%)", 10, 50, 15)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size/(100 - test_size), random_state=42)

# Sidebar for Model Parameters
st.sidebar.title("⚙️ Model Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
n_iterations = st.sidebar.slider("Number of Iterations", 100, 2000, 1000)
lambda_param = st.sidebar.slider("Regularization Strength (lambda)", 0.0, 1.0, 0.1)

# Feature Preprocessing
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
])

X_train_transformed = preprocessor.fit_transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

numeric_names = numeric_features
cat_names = []
for feature in categorical_features:
    categories = X[feature].dropna().unique()[1:]
    cat_names.extend([f"{feature}_{cat}" for cat in categories])
all_feature_names = numeric_names + cat_names

# L1LinearRegression
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
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.lambda_param / n_samples) * np.sign(self.weights)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# L2LinearRegression
class L2LinearRegression:
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
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (2 * self.lambda_param / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train Models
if st.button("Train L1 & L2 Models"):
    l1_model = L1LinearRegression(learning_rate, n_iterations, lambda_param)
    l2_model = L2LinearRegression(learning_rate, n_iterations, lambda_param)

    l1_model.fit(X_train_transformed, y_train)
    l2_model.fit(X_train_transformed, y_train)

    st.session_state['L1_model'] = {
        'model': l1_model,
        'feature_names': all_feature_names,
        'X_train': X_train_transformed,
        'y_train': y_train,
        'X_val': X_val_transformed,
        'y_val': y_val,
        'X_test': X_test_transformed,
        'y_test': y_test,
        'name': 'L1 (Lasso)'
    }

    st.session_state['L2_model'] = {
        'model': l2_model,
        'feature_names': all_feature_names,
        'X_train': X_train_transformed,
        'y_train': y_train,
        'X_val': X_val_transformed,
        'y_val': y_val,
        'X_test': X_test_transformed,
        'y_test': y_test,
        'name': 'L2 (Ridge)'
    }

    st.success("Both L1 and L2 models trained successfully.")
