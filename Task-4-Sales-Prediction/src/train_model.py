"""
Model Training Module
Sales Prediction Project — CodeAlpha Data Science Internship

Functions for training and comparing multiple regression models.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib
import os


def train_all_models(X_train, y_train):
    """Train multiple regression models and return them in a dictionary."""

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"Trained: {name}")

    return trained_models


def get_best_model(trained_models, X_test, y_test):
    """Compare all models and return the best one based on R² score."""

    best_score = -np.inf
    best_name = None
    best_model = None

    for name, model in trained_models.items():
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    print(f"\nBest Model: {best_name}")
    print(f"R² Score: {best_score:.4f}")

    return best_name, best_model, best_score


def save_model(model, filepath='../models/trained_model.pkl'):
    """Save the trained model to a file."""

    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath='../models/trained_model.pkl'):
    """Load a trained model from a file."""

    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


if __name__ == "__main__":
    # Quick test with sample data
    from data_preprocessing import load_data, clean_data, prepare_training_data

    data = load_data("../data/raw/Advertising.csv")
    cleaned = clean_data(data)
    X_train, X_test, y_train, y_test = prepare_training_data(cleaned)

    trained_models = train_all_models(X_train, y_train)
    best_name, best_model, best_score = get_best_model(trained_models, X_test, y_test)

    save_model(best_model)
    print("\nModel training complete!")
