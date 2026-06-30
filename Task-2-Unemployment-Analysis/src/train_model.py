import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os


def prepare_features(df):
    """Encode categorical columns and prepare features for training."""
    df_model = df.copy()
    
    # Encode categorical variables
    le_region = LabelEncoder()
    le_area = LabelEncoder()
    
    df_model['Region_Encoded'] = le_region.fit_transform(df_model['Region'])
    df_model['Area_Encoded'] = le_area.fit_transform(df_model['Area'])
    
    # Create COVID period flag
    df_model['Is_COVID'] = (df_model['Date'] >= pd.Timestamp('2020-04-01')).astype(int)
    
    # Define features and target
    features = ['Region_Encoded', 'Area_Encoded', 'Month', 'Year',
                'Labour_Participation_Rate', 'Is_COVID']
    target = 'Unemployment_Rate'
    
    X = df_model[features]
    y = df_model[target]
    
    return X, y, le_region, le_area


def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    """Train a Decision Tree Regressor."""
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, model_path, filename='random_forest_model.pkl'):
    """Save a trained model to disk."""
    os.makedirs(model_path, exist_ok=True)
    filepath = os.path.join(model_path, filename)
    
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    
    print(f'Model saved to {filepath}')


if __name__ == '__main__':
    # Load processed data
    df = pd.read_csv(os.path.join('..', 'data', 'processed', 'unemployment_india_cleaned.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Prepare features
    X, y, le_region, le_area = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # Save the best model
    save_model(rf_model, os.path.join('..', 'models'))
    
    print('All models trained successfully')
