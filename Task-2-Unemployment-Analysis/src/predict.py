import pickle
import pandas as pd
import numpy as np
import os


def load_model(model_path):
    """Load a trained model from disk."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def make_predictions(model, X):
    """Make predictions using the loaded model."""
    predictions = model.predict(X)
    return predictions


def display_results(y_actual, y_predicted, num_samples=10):
    """Display actual vs predicted results in a clean format."""
    results = pd.DataFrame({
        'Actual': y_actual[:num_samples],
        'Predicted': np.round(y_predicted[:num_samples], 2),
        'Difference': np.round(abs(y_actual[:num_samples] - y_predicted[:num_samples]), 2)
    })
    
    print('Sample Predictions')
    print('=' * 40)
    print(results.to_string(index=False))
    print()
    
    avg_error = np.mean(abs(y_actual - y_predicted))
    print(f'Average Prediction Error: {avg_error:.2f}%')


if __name__ == '__main__':
    # Load model
    model_path = os.path.join('..', 'models', 'random_forest_model.pkl')
    model = load_model(model_path)
    
    # Load test data
    df = pd.read_csv(os.path.join('..', 'data', 'processed', 'unemployment_india_cleaned.csv'))
    
    print('Model loaded successfully')
    print(f'Dataset shape: {df.shape}')
