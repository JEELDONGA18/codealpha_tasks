import pandas as pd
import os


def load_data(raw_data_path):
    """Load the raw unemployment datasets."""
    df1 = pd.read_csv(os.path.join(raw_data_path, 'Unemployment in India.csv'))
    df2 = pd.read_csv(os.path.join(raw_data_path, 'Unemployment_Rate_upto_11_2020.csv'))
    return df1, df2


def clean_data(df1, df2):
    """Clean both datasets — fix column names, parse dates, handle missing values."""
    
    # Strip whitespace from column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Rename columns for easier use
    rename_map = {
        'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
        'Estimated Employed': 'Employed',
        'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
    }
    df1 = df1.rename(columns=rename_map)
    df2 = df2.rename(columns=rename_map)
    
    # Handle duplicate 'Region' column in df2
    cols = list(df2.columns)
    if cols.count('Region') > 1:
        second_idx = cols.index('Region', cols.index('Region') + 1)
        cols[second_idx] = 'Geographic_Region'
        df2.columns = cols
    
    # Parse dates
    df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True)
    df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=True)
    
    # Drop rows where unemployment rate is missing
    df1 = df1.dropna(subset=['Unemployment_Rate'])
    df2 = df2.dropna(subset=['Unemployment_Rate'])
    
    # Drop duplicates
    df1 = df1.drop_duplicates()
    df2 = df2.drop_duplicates()
    
    # Reset index
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    # Add Month and Year columns
    df1['Month'] = df1['Date'].dt.month
    df1['Year'] = df1['Date'].dt.year
    df2['Month'] = df2['Date'].dt.month
    df2['Year'] = df2['Date'].dt.year
    
    return df1, df2


def save_processed_data(df1, df2, processed_path):
    """Save cleaned dataframes to the processed folder."""
    os.makedirs(processed_path, exist_ok=True)
    
    df1.to_csv(os.path.join(processed_path, 'unemployment_india_cleaned.csv'), index=False)
    df2.to_csv(os.path.join(processed_path, 'unemployment_rate_cleaned.csv'), index=False)
    
    print('Processed data saved successfully')


if __name__ == '__main__':
    raw_path = os.path.join('..', 'data', 'raw')
    processed_path = os.path.join('..', 'data', 'processed')
    
    df1, df2 = load_data(raw_path)
    df1, df2 = clean_data(df1, df2)
    save_processed_data(df1, df2, processed_path)
    
    print(f'Dataset 1: {df1.shape}')
    print(f'Dataset 2: {df2.shape}')
