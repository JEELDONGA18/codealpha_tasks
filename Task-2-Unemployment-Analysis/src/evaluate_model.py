import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os


def calculate_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4)
    }


def print_metrics(name, metrics):
    """Print evaluation metrics in a clean format."""
    print(f'--- {name} ---')
    print(f'MAE:  {metrics["MAE"]}')
    print(f'MSE:  {metrics["MSE"]}')
    print(f'RMSE: {metrics["RMSE"]}')
    print(f'R²:   {metrics["R2"]}')
    print()


def plot_actual_vs_predicted(y_true, y_pred, model_name, save_path=None):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='teal', edgecolors='black', linewidth=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    plt.title(f'Actual vs Predicted ({model_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Unemployment Rate (%)')
    plt.ylabel('Predicted Unemployment Rate (%)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_residuals(y_true, y_pred, model_name, save_path=None):
    """Plot residuals to check model performance."""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='salmon', edgecolors='black', linewidth=0.5)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.title(f'Residual Plot ({model_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Unemployment Rate (%)')
    plt.ylabel('Residuals')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compare_models(results_dict, save_path=None):
    """Create a bar chart comparing R² scores of all models."""
    models = list(results_dict.keys())
    r2_scores = [results_dict[m]['R2'] for m in models]
    
    colors = ['#3498db', '#e67e22', '#2ecc71']
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, r2_scores, color=colors[:len(models)], edgecolor='black')
    plt.title('Model Comparison — R² Score', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{score:.4f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
