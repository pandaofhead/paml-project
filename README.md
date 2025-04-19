# Sleep Disorder Prediction using Machine Learning

This Streamlit application predicts sleep disorders using health and lifestyle data through machine learning models. The current implementation includes L1 Linear Regression, with plans to add L2 Linear Regression in the future.

## Features

- L1 Linear Regression implementation from scratch
- Model evaluation using MAE, RMSE, and R² metrics
- Interactive parameter tuning through Streamlit interface
- Data preprocessing and feature scaling
- Visualization of model performance

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## Model Parameters

- Learning Rate: Controls the step size in gradient descent
- Number of Iterations: Number of training iterations
- L1 Regularization Parameter: Controls the strength of L1 regularization

## Data Split

The dataset is split into:
- 70% Training
- 15% Validation
- 15% Testing

## Future Improvements

- Implementation of L2 Linear Regression
- Addition of real sleep disorder dataset
- Enhanced feature engineering
- Model comparison and selection based on weighted scoring system 