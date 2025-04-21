# Sleep Health and Lifestyle Analysis

This Streamlit application analyzes sleep health data and predicts sleep quality using machine learning models. The application implements both L1 (Lasso) and L2 (Ridge) regularized linear regression models from scratch.

## Project Structure

- `Introduction.py`: Main entry point for the Streamlit application
- `pages/`: Directory containing the application pages
  - `0_Explore_&_Preprocess.py`: Data exploration and preprocessing
  - `1_Train_Model.py`: Model training with hyperparameter tuning
  - `2_Test_Model.py`: Model evaluation and visualization
- `helper_functions.py`: Utility functions for the application
- `Sleep_health_and_lifestyle_dataset.csv`: Original dataset
- `Preprocessed_Sleep_Health_Dataset.xlsx`: Preprocessed dataset
- `PAML data preprocessing.ipynb`: Jupyter notebook for data preprocessing
- `assets/`: Directory containing images and other assets

## Features

- **Data Exploration and Preprocessing**:
  - Interactive data visualization (histograms, boxplots, scatterplots, correlation heatmaps)
  - Feature engineering and preprocessing (one-hot encoding, scaling, outlier handling)
  - PCA for dimensionality reduction

- **Model Training**:
  - Implementation of L1 (Lasso) and L2 (Ridge) regularized linear regression from scratch
  - Interactive hyperparameter tuning (learning rate, iterations, regularization strength)
  - Train/validation/test split with customizable proportions

- **Model Evaluation**:
  - Performance metrics: MAE, RMSE, and R²
  - Visualization of actual vs. predicted values
  - Feature importance analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run Introduction.py
```

2. Navigate through the application:
   - Start with the Introduction page to understand the project
   - Go to "Explore & Preprocess" to analyze and preprocess the dataset
   - Train models in the "Train Model" page
   - Evaluate models in the "Test Model" page

## Dataset

The application uses the Sleep Health and Lifestyle Dataset, which contains information about:
- Sleep duration and quality
- Physical activity levels
- Stress levels
- BMI categories
- Blood pressure
- Heart rate
- Daily steps
- Sleep disorders (insomnia, sleep apnea)

## Model Parameters

- **Learning Rate**: Controls the step size in gradient descent (default: 0.01)
- **Number of Iterations**: Number of training iterations (default: 1000)
- **Regularization Strength (lambda)**: Controls the strength of L1/L2 regularization (default: 0.1)

## Data Split

The dataset is split into:
- Training set (default: 55%)
- Validation set (default: 15%)
- Test set (default: 30%)
