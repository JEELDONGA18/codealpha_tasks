"""
Data Preprocessing Module for Car Price Prediction.

This module handles loading raw car data, cleaning it,
engineering new features, and saving the processed data.
"""

import os
import pandas as pd


def load_data(filepath):
    """Read a CSV file and return a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    data = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {data.shape}")
    return data


def clean_data(df):
    """Clean the DataFrame by removing duplicates and missing values.

    Args:
        df: The raw pandas DataFrame.

    Returns:
        A cleaned pandas DataFrame.
    """
    # Store original shape for reporting
    original_shape = df.shape

    # Remove duplicate rows
    df = df.drop_duplicates()
    duplicates_removed = original_shape[0] - df.shape[0]
    print(f"Duplicates removed: {duplicates_removed}")

    # Remove rows with missing values
    rows_before = df.shape[0]
    df = df.dropna()
    nan_rows_removed = rows_before - df.shape[0]
    print(f"Rows with NaN removed: {nan_rows_removed}")

    # Verify numeric columns have correct data types
    numeric_columns = ["Year", "Selling_Price", "Present_Price", "Driven_kms", "Owner"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Drop any new NaN rows created by type conversion
    df = df.dropna()

    # Reset index after cleaning
    df = df.reset_index(drop=True)

    print(f"Cleaned data shape: {df.shape}")
    return df


def extract_brand(car_name):
    """Extract the brand name from a full car name string.

    Takes the first word of the car name and converts it to lowercase.

    Args:
        car_name: A string containing the full car name (e.g., 'Honda City').

    Returns:
        The brand name in lowercase (e.g., 'honda').
    """
    brand = car_name.split(" ")[0].lower()
    return brand


def engineer_features(df):
    """Create new features from existing columns for model training.

    New features created:
        - Car_Age: Calculated as 2025 minus the Year.
        - Brand: Extracted from Car_Name using the first word.
        - Fuel_Type_Encoded: Petrol=0, Diesel=1, CNG=2.
        - Selling_Type_Encoded: Dealer=0, Individual=1.
        - Transmission_Encoded: Manual=0, Automatic=1.

    Args:
        df: A cleaned pandas DataFrame.

    Returns:
        The DataFrame with new engineered feature columns added.
    """
    # Calculate car age
    df["Car_Age"] = 2025 - df["Year"]
    print("Created Car_Age feature.")

    # Extract brand from car name
    df["Brand"] = df["Car_Name"].apply(extract_brand)
    print("Extracted Brand feature.")

    # Encode Fuel_Type
    fuel_type_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    df["Fuel_Type_Encoded"] = df["Fuel_Type"].map(fuel_type_mapping)
    print("Encoded Fuel_Type.")

    # Encode Selling_type
    selling_type_mapping = {"Dealer": 0, "Individual": 1}
    df["Selling_Type_Encoded"] = df["Selling_type"].map(selling_type_mapping)
    print("Encoded Selling_type.")

    # Encode Transmission
    transmission_mapping = {"Manual": 0, "Automatic": 1}
    df["Transmission_Encoded"] = df["Transmission"].map(transmission_mapping)
    print("Encoded Transmission.")

    return df


def save_processed_data(df, filepath):
    """Save the processed DataFrame to a CSV file.

    Creates the output directory if it does not exist.

    Args:
        df: The processed pandas DataFrame.
        filepath: Path where the CSV file will be saved.
    """
    # Create directory if it doesn't exist
    output_directory = os.path.dirname(filepath)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"Processed data saved to: {filepath}")


if __name__ == "__main__":
    # Set up file paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, "data", "raw", "car_data.csv")
    processed_data_path = os.path.join(project_root, "data", "processed", "car_data_processed.csv")

    print("=" * 60)
    print("Car Price Prediction - Data Preprocessing")
    print("=" * 60)

    # Step 1: Load the raw data
    print("\n--- Step 1: Loading Data ---")
    car_data = load_data(raw_data_path)

    # Step 2: Clean the data
    print("\n--- Step 2: Cleaning Data ---")
    car_data = clean_data(car_data)

    # Step 3: Engineer features
    print("\n--- Step 3: Engineering Features ---")
    car_data = engineer_features(car_data)

    # Step 4: Save processed data
    print("\n--- Step 4: Saving Processed Data ---")
    save_processed_data(car_data, processed_data_path)

    # Print summary
    print("\n--- Summary ---")
    print(f"Final dataset shape: {car_data.shape}")
    print(f"Columns: {list(car_data.columns)}")
    print(f"\nFirst 5 rows:")
    print(car_data.head())
    print(f"\nData types:\n{car_data.dtypes}")
    print(f"\nBasic statistics:\n{car_data.describe()}")
    print("\nPreprocessing complete!")
