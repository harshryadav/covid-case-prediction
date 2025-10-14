import pandas as pd
import numpy as np
from pathlib import Path

# Set paths
data_dir = Path('../../data')
output_dir = Path('./training')

# Read the datasets
cases_df = pd.read_csv(data_dir / 'time_series_covid19_confirmed_US.csv')
vaccine_df = pd.read_csv(data_dir / 'time_series_covid19_vaccine_us.csv')

def process_cases_data(df):
    # Get date columns (excluding metadata columns)
    date_columns = df.columns[11:]  # First 11 columns are metadata
    
    # Melt the dataframe to convert to long format
    melted_df = df.melt(
        id_vars=['Province_State', 'Admin2', 'Lat', 'Long_'],
        value_vars=date_columns,
        var_name='Date',
        value_name='Confirmed_Cases'
    )
    
    # Convert date strings to datetime
    melted_df['Date'] = pd.to_datetime(melted_df['Date'])
    
    # Group by state and date
    state_cases = melted_df.groupby(['Province_State', 'Date'])['Confirmed_Cases'].sum().reset_index()
    return state_cases

def process_vaccine_data(df):
    # Convert date strings to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Group by state and date
    state_vaccines = df.groupby(['Province_State', 'Date'])['People_fully_vaccinated'].sum().reset_index()
    return state_vaccines

# Process both datasets
cases_ts = process_cases_data(cases_df)
vaccine_ts = process_vaccine_data(vaccine_df)

# Merge the datasets
merged_ts = pd.merge(
    cases_ts,
    vaccine_ts,
    on=['Province_State', 'Date'],
    how='outer'
)

# Sort by state and date
merged_ts = merged_ts.sort_values(['Province_State', 'Date'])

# Fill missing values with forward fill within each state
merged_ts = merged_ts.groupby('Province_State').fillna(method='ffill').reset_index()

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Save the processed time series data
merged_ts.to_csv(output_dir / 'covid_time_series.csv', index=False)
print("Time series data has been created and saved to training/covid_time_series.csv")