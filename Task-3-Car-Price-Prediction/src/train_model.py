"""
Model Training Module for Car Price Prediction.

This module handles preparing data for training, training multiple
regression models, evaluating them, and saving the best performer.
"""

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

from data_preprocessing import load_data, clean_data, engineer_features


def prepare_data(df):
    """Split the DataFrame into training and testing sets.

    Selects the relevant features and target variable, then performs
    an 80/20 train-test split.

    Args:
        df: A pandas DataFrame with engineered features.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """
    # Define the feature columns
    feature_columns = [
        "Present_Price",
        "Driven_kms",
        "Car_Age",
        "Fuel_Type_Encoded",
        "Selling_Type_Encoded",
        "Transmission_Encoded",
        "Owner"
    ]

    # Separate features and target
    X = df[feature_columns]
    y = df["Selling_Price"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def train_all_models(X_train, y_train):
    """Train multiple regression models on the training data.

    Models trained:
        - Linear Regression
        - Decision Tree Regressor
        - Random Forest Regressor (100 estimators)
        - Gradient Boosting Regressor (100 estimators)

    Args:
        X_train: Training feature data.
        y_train: Training target data.

    Returns:
        A dictionary mapping model names to trained model objects.
    """
    # Create model instances
    linear_regression = LinearRegression()
    decision_tree = DecisionTreeRegressor(random_state=42)
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    gradient_boosting = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Store models in a dictionary
    models = {
        "Linear Regression": linear_regression,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting
    }

    # Train each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        print(f"  {model_name} trained successfully.")

    return models


def save_model(model, filepath):
    """Save a trained model to disk using joblib.

    Creates the output directory if it does not exist.

    Args:
        model: The trained model object to save.
        filepath: Path where the model file will be saved.
    """
    # Create directory if it doesn't exist
    output_directory = os.path.dirname(filepath)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


if __name__ == "__main__":
    # Set up file paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, "data", "raw", "car_data.csv")
    model_save_path = os.path.join(project_root, "models", "trained_model.pkl")

    print("=" * 60)
    print("Car Price Prediction - Model Training")
    print("=" * 60)

    # Step 1: Load and preprocess the data
    print("\n--- Step 1: Loading and Preprocessing Data ---")
    car_data = load_data(raw_data_path)
    car_data = clean_data(car_data)
    car_data = engineer_features(car_data)

    # Step 2: Prepare training and testing sets
    print("\n--- Step 2: Preparing Train/Test Split ---")
    X_train, X_test, y_train, y_test = prepare_data(car_data)

    # Step 3: Train all models
    print("\n--- Step 3: Training Models ---")
    trained_models = train_all_models(X_train, y_train)

    # Step 4: Evaluate each model and find the best one
    print("\n--- Step 4: Evaluating Models ---")
    best_model_name = None
    best_model = None
    best_r2_score = -1.0

    for model_name, model in trained_models.items():
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)
        print(f"  {model_name}: R² Score = {score:.4f}")

        # Track the best model
        if score > best_r2_score:
            best_r2_score = score
            best_model_name = model_name
            best_model = model

    # Step 5: Save the best model
    print(f"\n--- Step 5: Saving Best Model ---")
    print(f"Best model: {best_model_name} (R² = {best_r2_score:.4f})")
    save_model(best_model, model_save_path)

    print("\nModel training complete!")
