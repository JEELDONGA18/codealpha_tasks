"""
Model Evaluation Module
Sales Prediction Project — CodeAlpha Data Science Internship

Functions for evaluating models and generating comparison metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a model and return key regression metrics."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results = {
        'Model': model_name,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R² Score': round(r2, 4)
    }

    return results


def compare_models(trained_models, X_test, y_test):
    """Evaluate all models and return a comparison DataFrame."""

    all_results = []

    for name, model in trained_models.items():
        results = evaluate_model(model, X_test, y_test, model_name=name)
        all_results.append(results)

    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('R² Score', ascending=False)
    comparison_df = comparison_df.reset_index(drop=True)

    return comparison_df


def plot_actual_vs_predicted(y_test, predictions, model_name="Model", save_path=None):
    """Create an Actual vs Predicted scatter plot."""

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test, predictions, alpha=0.6, color='#2196F3', edgecolors='white', s=80)

    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Sales', fontsize=12)
    ax.set_ylabel('Predicted Sales', fontsize=12)
    ax.set_title(f'Actual vs Predicted Sales — {model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_residuals(y_test, predictions, model_name="Model", save_path=None):
    """Create a residual plot to check model assumptions."""

    residuals = y_test - predictions

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(predictions, residuals, alpha=0.6, color='#FF7043', edgecolors='white', s=80)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Predicted Sales', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f'Residual Plot — {model_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_feature_importance(model, feature_names, model_name="Model", save_path=None):
    """Plot feature importance as a horizontal bar chart."""

    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print(f"Cannot extract feature importance from {model_name}")
        return

    # Sort by importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#66BB6A', '#42A5F5', '#FFA726']
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='white', height=0.5)

    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance — {model_name}', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, value in zip(bars, importance_df['Importance']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{value:.4f}', va='center', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data, prepare_training_data
    from train_model import train_all_models, get_best_model

    data = load_data("../data/raw/Advertising.csv")
    cleaned = clean_data(data)
    X_train, X_test, y_train, y_test = prepare_training_data(cleaned)

    trained_models = train_all_models(X_train, y_train)
    comparison = compare_models(trained_models, X_test, y_test)
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))

    best_name, best_model, _ = get_best_model(trained_models, X_test, y_test)
    predictions = best_model.predict(X_test)

    plot_actual_vs_predicted(y_test, predictions, best_name)
    plot_residuals(y_test, predictions, best_name)
    plot_feature_importance(best_model, X_train.columns.tolist(), best_name)
