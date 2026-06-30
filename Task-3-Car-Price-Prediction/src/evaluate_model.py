"""
Model Evaluation Module for Car Price Prediction.

This module provides functions to evaluate regression models,
compare multiple models, and generate diagnostic visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return key regression metrics.

    Args:
        model: A trained regression model.
        X_test: Test feature data.
        y_test: Test target data.

    Returns:
        A dictionary containing MAE, MSE, RMSE, and R2_Score.
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2_Score": round(r2, 4)
    }

    return metrics


def compare_models(models, X_test, y_test):
    """Evaluate multiple models and return a comparison DataFrame.

    Args:
        models: A dictionary of {model_name: trained_model}.
        X_test: Test feature data.
        y_test: Test target data.

    Returns:
        A pandas DataFrame with each model's evaluation metrics.
    """
    results = []

    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = model_name
        results.append(metrics)

    # Create DataFrame and reorder columns
    comparison_df = pd.DataFrame(results)
    column_order = ["Model", "MAE", "MSE", "RMSE", "R2_Score"]
    comparison_df = comparison_df[column_order]

    # Sort by R2_Score in descending order (best model first)
    comparison_df = comparison_df.sort_values(by="R2_Score", ascending=False)
    comparison_df = comparison_df.reset_index(drop=True)

    return comparison_df


def plot_actual_vs_predicted(y_test, predictions, save_path=None):
    """Create a scatter plot of actual vs predicted values.

    Includes a diagonal reference line showing perfect predictions.

    Args:
        y_test: Actual target values.
        predictions: Predicted values from the model.
        save_path: Optional path to save the plot image.
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot of actual vs predicted
    plt.scatter(y_test, predictions, color="#3498db", alpha=0.6, edgecolors="black", linewidth=0.5)

    # Add diagonal reference line for perfect predictions
    min_value = min(min(y_test), min(predictions))
    max_value = max(max(y_test), max(predictions))
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", linewidth=2)

    plt.xlabel("Actual Selling Price (Lakhs)", fontsize=12)
    plt.ylabel("Predicted Selling Price (Lakhs)", fontsize=12)
    plt.title("Actual vs Predicted Car Prices", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        # Create directory if it doesn't exist
        output_directory = os.path.dirname(save_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_residuals(y_test, predictions, save_path=None):
    """Create a scatter plot of residuals (actual minus predicted).

    Includes a horizontal reference line at zero.

    Args:
        y_test: Actual target values.
        predictions: Predicted values from the model.
        save_path: Optional path to save the plot image.
    """
    residuals = y_test - predictions

    plt.figure(figsize=(10, 6))

    # Scatter plot of residuals
    plt.scatter(predictions, residuals, color="#e74c3c", alpha=0.6, edgecolors="black", linewidth=0.5)

    # Add horizontal reference line at zero
    plt.axhline(y=0, color="black", linestyle="--", linewidth=2)

    plt.xlabel("Predicted Selling Price (Lakhs)", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
    plt.title("Residual Plot", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        output_directory = os.path.dirname(save_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_feature_importance(model, feature_names, save_path=None):
    """Create a horizontal bar chart showing feature importances.

    Works with models that have a feature_importances_ attribute
    (e.g., Decision Tree, Random Forest, Gradient Boosting).

    Args:
        model: A trained model with feature_importances_ attribute.
        feature_names: A list of feature names.
        save_path: Optional path to save the plot image.
    """
    importances = model.feature_importances_

    # Create a DataFrame for easy sorting
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    importance_df = importance_df.sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 6))

    # Horizontal bar chart
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5
    )

    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("Feature Importance", fontsize=14)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        output_directory = os.path.dirname(save_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Car Price Prediction - Model Evaluation Module")
    print("=" * 60)
    print()
    print("This module provides evaluation and visualization functions:")
    print("  - evaluate_model(): Compute MAE, MSE, RMSE, R2 Score")
    print("  - compare_models(): Compare multiple models side by side")
    print("  - plot_actual_vs_predicted(): Scatter plot of actual vs predicted")
    print("  - plot_residuals(): Residual analysis plot")
    print("  - plot_feature_importance(): Feature importance bar chart")
    print()
    print("Import this module from train_model.py or a notebook to use.")
    print("Example:")
    print("  from evaluate_model import evaluate_model, compare_models")
    print("  metrics = evaluate_model(model, X_test, y_test)")
    print("  print(metrics)")
