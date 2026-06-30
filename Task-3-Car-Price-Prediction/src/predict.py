"""
Prediction Module for Car Price Prediction.

This module handles loading a saved trained model and making
predictions on new car data.
"""

import os
import joblib
import pandas as pd


def load_model(filepath):
    """Load a trained model from disk using joblib.

    Args:
        filepath: Path to the saved model file (.pkl).

    Returns:
        The loaded trained model object.
    """
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def predict_price(model, car_features):
    """Predict the selling price for a car given its features.

    Args:
        model: A trained regression model.
        car_features: A dictionary containing the car's features.
            Expected keys: Present_Price, Driven_kms, Car_Age,
            Fuel_Type_Encoded, Selling_Type_Encoded,
            Transmission_Encoded, Owner.

    Returns:
        The predicted selling price as a float (in Lakhs).
    """
    # Define the expected column order for the model
    column_order = [
        "Present_Price",
        "Driven_kms",
        "Car_Age",
        "Fuel_Type_Encoded",
        "Selling_Type_Encoded",
        "Transmission_Encoded",
        "Owner"
    ]

    # Create a single-row DataFrame with the correct column order
    input_data = pd.DataFrame([car_features], columns=column_order)

    # Make the prediction
    predicted_price = model.predict(input_data)

    # Return as a plain float
    return float(predicted_price[0])


if __name__ == "__main__":
    # Set up file path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "trained_model.pkl")

    print("=" * 60)
    print("Car Price Prediction - Price Predictor")
    print("=" * 60)

    # Load the trained model
    print("\nLoading trained model...")
    trained_model = load_model(model_path)

    # Create sample car features
    sample_car = {
        "Present_Price": 9.85,
        "Driven_kms": 6900,
        "Car_Age": 8,
        "Fuel_Type_Encoded": 0,
        "Selling_Type_Encoded": 0,
        "Transmission_Encoded": 0,
        "Owner": 0
    }

    print("\nSample Car Features:")
    for feature_name, feature_value in sample_car.items():
        print(f"  {feature_name}: {feature_value}")

    # Predict the price
    predicted_selling_price = predict_price(trained_model, sample_car)

    print(f"\nPredicted Selling Price: {predicted_selling_price:.2f} Lakhs")
    print("\nPrediction complete!")
