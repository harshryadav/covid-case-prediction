"""
Data Preprocessing Utilities for COVID-19 Time Series

This module provides functions to aggregate, clean, and merge COVID-19 data
for time series forecasting with GluonTS.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def aggregate_to_national(df: pd.DataFrame, data_type: str = 'cases') -> pd.DataFrame:
    """
    Aggregate county-level data to national level.
    
    Args:
        df: DataFrame with county-level time series (JHU CSSE format)
        data_type: Type of data ('cases' or 'deaths') for column naming
        
    Returns:
        DataFrame with Date and national-level daily/cumulative values
        
    Example:
        >>> national_cases = aggregate_to_national(cases_df, data_type='cases')
        >>> print(national_cases.columns)
        ['Date', 'Cumulative_Cases', 'Daily_Cases', 'Daily_Cases_MA7']
    """
    # Deaths file has 12 metadata columns (includes Population), cases has 11
    skip_cols = 12 if data_type == 'deaths' else 11
    date_columns = df.columns[skip_cols:]
    
    # Sum across all counties for each date
    national_cumulative = df[date_columns].sum()
    
    # Create DataFrame with appropriate column names
    prefix = 'Cases' if data_type == 'cases' else 'Deaths'
    result_df = pd.DataFrame({
        'Date': pd.to_datetime(national_cumulative.index),
        f'Cumulative_{prefix}': national_cumulative.values
    })
    
    # Calculate daily new values
    result_df[f'Daily_{prefix}'] = result_df[f'Cumulative_{prefix}'].diff().fillna(0)
    
    # Calculate 7-day moving average
    result_df[f'Daily_{prefix}_MA7'] = result_df[f'Daily_{prefix}'].rolling(window=7).mean()
    
    # Remove negative values (reporting corrections)
    result_df[f'Daily_{prefix}'] = result_df[f'Daily_{prefix}'].clip(lower=0)
    result_df[f'Daily_{prefix}_MA7'] = result_df[f'Daily_{prefix}_MA7'].clip(lower=0)
    
    return result_df


def extract_national_mobility(mobility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract national-level mobility data from Google Mobility Reports.
    
    Args:
        mobility_df: DataFrame with mobility data (multiple geographic levels)
        
    Returns:
        DataFrame with national mobility trends (6 categories)
        
    Example:
        >>> national_mobility = extract_national_mobility(mobility_df)
        >>> print(national_mobility.columns)
        ['date', 'state', 'county', 'retail and recreation', ...]
    """
    # Filter for national level (state='Total', county='Total')
    national = mobility_df[
        (mobility_df['state'] == 'Total') & 
        (mobility_df['county'] == 'Total')
    ].copy()
    
    # Sort by date
    national = national.sort_values('date').reset_index(drop=True)
    
    return national


def merge_all_data(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    mobility_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge cases, deaths, and mobility data into a single DataFrame.
    
    This function:
    - Merges on date
    - Fills missing values appropriately
    - Calculates Case Fatality Ratio (CFR)
    
    Args:
        cases_df: National cases DataFrame
        deaths_df: National deaths DataFrame
        mobility_df: National mobility DataFrame
        
    Returns:
        Merged DataFrame with all features
        
    Example:
        >>> merged = merge_all_data(national_cases, national_deaths, national_mobility)
        >>> print(merged.shape)
        (1143, 16)
    """
    # Ensure date columns are datetime
    cases_df['Date'] = pd.to_datetime(cases_df['Date'])
    deaths_df['Date'] = pd.to_datetime(deaths_df['Date'])
    mobility_df['date'] = pd.to_datetime(mobility_df['date'])
    
    # Merge cases and deaths
    merged = pd.merge(
        cases_df,
        deaths_df[['Date', 'Daily_Deaths', 'Daily_Deaths_MA7', 'Cumulative_Deaths']],
        on='Date',
        how='left'
    )
    
    # Merge with mobility
    merged = pd.merge(
        merged,
        mobility_df,
        left_on='Date',
        right_on='date',
        how='left'
    )
    
    # Drop duplicate date column
    merged = merged.drop('date', axis=1)
    
    # Forward fill missing mobility values
    mobility_cols = [
        'retail and recreation', 'grocery and pharmacy', 
        'parks', 'transit stations', 'workplaces', 'residential'
    ]
    merged[mobility_cols] = merged[mobility_cols].fillna(method='ffill')
    
    # Fill missing deaths data (early pandemic period)
    death_cols = ['Daily_Deaths', 'Daily_Deaths_MA7', 'Cumulative_Deaths']
    merged[death_cols] = merged[death_cols].fillna(0)
    
    # Calculate case fatality ratio (useful feature)
    merged['CFR'] = (
        merged['Cumulative_Deaths'] / merged['Cumulative_Cases'].replace(0, 1)
    ) * 100
    merged['CFR'] = merged['CFR'].fillna(0).clip(0, 100)  # Cap at 100%
    
    return merged


def create_train_test_split(
    df: pd.DataFrame,
    test_days: int = 14
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.
    
    Args:
        df: Full dataset
        test_days: Number of days to reserve for testing
        
    Returns:
        Tuple of (train_df, test_df)
        
    Example:
        >>> train_df, test_df = create_train_test_split(merged_df, test_days=14)
        >>> print(f"Train: {len(train_df)} days, Test: {len(test_df)} days")
    """
    # Handle NaN values in the target column
    df = df.dropna(subset=['Daily_Cases_MA7'])
    
    # Split into train and test
    split_idx = len(df) - test_days
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


def preprocess_pipeline(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    mobility_df: pd.DataFrame,
    test_days: int = 14
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline from raw data to train/test split.
    
    Args:
        cases_df: Raw JHU cases data
        deaths_df: Raw JHU deaths data
        mobility_df: Raw Google mobility data
        test_days: Number of days for test set
        
    Returns:
        Tuple of (merged_df, train_df, test_df)
        
    Example:
        >>> merged, train, test = preprocess_pipeline(cases_df, deaths_df, mobility_df)
        >>> print(f"Merged: {merged.shape}, Train: {train.shape}, Test: {test.shape}")
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Step 1: Aggregate to national level
    print("\n[Step 1/4] Aggregating cases to national level...")
    national_cases = aggregate_to_national(cases_df, data_type='cases')
    print(f"  ✓ National cases: {len(national_cases)} days")
    
    print("[Step 2/4] Aggregating deaths to national level...")
    national_deaths = aggregate_to_national(deaths_df, data_type='deaths')
    print(f"  ✓ National deaths: {len(national_deaths)} days")
    
    # Step 2: Extract mobility
    print("[Step 3/4] Extracting national mobility data...")
    national_mobility = extract_national_mobility(mobility_df)
    print(f"  ✓ National mobility: {len(national_mobility)} days")
    
    # Step 3: Merge all data
    print("[Step 4/4] Merging all datasets...")
    merged_df = merge_all_data(national_cases, national_deaths, national_mobility)
    print(f"  ✓ Merged data: {len(merged_df)} days, {len(merged_df.columns)} features")
    
    # Step 4: Train/test split
    print(f"\nCreating train/test split ({test_days} days for testing)...")
    train_df, test_df = create_train_test_split(merged_df, test_days=test_days)
    print(f"  ✓ Train: {len(train_df)} days")
    print(f"  ✓ Test: {len(test_df)} days")
    
    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Date range: {merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}")
    print(f"Features: {list(merged_df.columns[:7])} + mobility (6) + CFR")
    print("=" * 70 + "\n")
    
    return merged_df, train_df, test_df

