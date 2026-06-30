"""
Data Preprocessing Module
Sales Prediction Project — CodeAlpha Data Science Internship

Functions for loading, cleaning, and preparing the advertising dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """Load the advertising dataset from a CSV file."""
    data = pd.read_csv(filepath)

    # Drop the unnamed index column if it exists
    if data.columns[0] == '' or 'Unnamed' in str(data.columns[0]):
        data = data.drop(columns=data.columns[0])

    return data


def clean_data(data):
    """Handle missing values, duplicates, and data types."""
    cleaned = data.copy()

    # Remove duplicate rows
    duplicates_before = cleaned.duplicated().sum()
    cleaned = cleaned.drop_duplicates()

    if duplicates_before > 0:
        print(f"Removed {duplicates_before} duplicate rows")

    # Handle missing values
    missing = cleaned.isnull().sum().sum()
    if missing > 0:
        # Fill numerical columns with median
        for column in cleaned.select_dtypes(include=[np.number]).columns:
            if cleaned[column].isnull().sum() > 0:
                median_value = cleaned[column].median()
                cleaned[column] = cleaned[column].fillna(median_value)
                print(f"Filled missing values in '{column}' with median: {median_value}")
    else:
        print("No missing values found")

    return cleaned


def engineer_features(data):
    """Create meaningful features from the existing data."""
    enhanced = data.copy()

    # Total advertising budget across all channels
    enhanced['Total_Budget'] = enhanced['TV'] + enhanced['Radio'] + enhanced['Newspaper']

    # Individual channel contribution (percentage)
    enhanced['TV_Share'] = enhanced['TV'] / enhanced['Total_Budget'] * 100
    enhanced['Radio_Share'] = enhanced['Radio'] / enhanced['Total_Budget'] * 100
    enhanced['Newspaper_Share'] = enhanced['Newspaper'] / enhanced['Total_Budget'] * 100

    # TV and Radio interaction (these two often work together)
    enhanced['TV_Radio_Interaction'] = enhanced['TV'] * enhanced['Radio']

    # Budget category based on total spend
    budget_bins = [0, 100, 200, 300, 500]
    budget_labels = ['Low', 'Medium', 'High', 'Very High']
    enhanced['Budget_Category'] = pd.cut(
        enhanced['Total_Budget'],
        bins=budget_bins,
        labels=budget_labels,
        include_lowest=True
    )

    return enhanced


def prepare_training_data(data, target_column='Sales', test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""

    # Select features for modeling (only numerical, exclude target and derived categories)
    feature_columns = ['TV', 'Radio', 'Newspaper']
    
    X = data[feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick test
    data = load_data("../data/raw/Advertising.csv")
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

    cleaned = clean_data(data)
    enhanced = engineer_features(cleaned)
    print(f"After feature engineering: {enhanced.shape[1]} columns")

    X_train, X_test, y_train, y_test = prepare_training_data(enhanced)
    print("Data preparation complete!")
