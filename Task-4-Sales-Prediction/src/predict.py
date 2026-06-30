"""
Prediction Module
Sales Prediction Project — CodeAlpha Data Science Internship

Functions for making sales predictions using the trained model.
"""

import pandas as pd
import joblib
import os


def load_trained_model(filepath='../models/trained_model.pkl'):
    """Load the trained model from disk."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No trained model found at: {filepath}")

    model = joblib.load(filepath)
    return model


def make_prediction(model, tv_budget, radio_budget, newspaper_budget):
    """Predict sales for given advertising budgets."""

    input_data = pd.DataFrame({
        'TV': [tv_budget],
        'Radio': [radio_budget],
        'Newspaper': [newspaper_budget]
    })

    predicted_sales = model.predict(input_data)[0]

    return round(predicted_sales, 2)


def predict_multiple(model, budget_data):
    """Predict sales for multiple advertising scenarios.
    
    budget_data: list of dictionaries with 'TV', 'Radio', 'Newspaper' keys
    """

    input_df = pd.DataFrame(budget_data)
    predictions = model.predict(input_df)

    input_df['Predicted_Sales'] = predictions.round(2)

    return input_df


if __name__ == "__main__":
    print("=" * 55)
    print("  Sales Prediction Tool")
    print("  CodeAlpha Data Science Internship — Task 4")
    print("=" * 55)

    try:
        model = load_trained_model()
        print("\nModel loaded successfully!\n")
    except FileNotFoundError:
        print("\nError: No trained model found.")
        print("Please run the notebook first to train and save a model.")
        exit()

    print("Enter advertising budgets (in thousands $):\n")

    try:
        tv = float(input("  TV Budget: $"))
        radio = float(input("  Radio Budget: $"))
        newspaper = float(input("  Newspaper Budget: $"))
    except ValueError:
        print("\nError: Please enter valid numbers.")
        exit()

    predicted_sales = make_prediction(model, tv, radio, newspaper)

    print(f"\n{'=' * 55}")
    print(f"  Predicted Sales: {predicted_sales} thousand units")
    print(f"{'=' * 55}")

    total_budget = tv + radio + newspaper
    print(f"\n  Total Ad Budget: ${total_budget:.1f}k")
    print(f"  Expected ROI: {predicted_sales / total_budget:.2f} units per $1k spent" if total_budget > 0 else "")
