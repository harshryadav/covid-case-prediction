"""
GluonTS Data Preparation Utilities

This module provides functions to prepare COVID-19 data for GluonTS models.
It handles data loading, formatting, and conversion to GluonTS ListDataset format.

The actual model training, testing, and prediction are done in the notebook files.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# GluonTS imports
from gluonts.dataset.common import ListDataset


def create_gluonts_dataset(
    df: pd.DataFrame,
    target_column: str,
    freq: str = 'D',
    prediction_length: int = 14,
    past_feat_columns: Optional[List[str]] = None
) -> ListDataset:
    """
    Convert pandas DataFrame to GluonTS ListDataset format.
    
    This is the main data preparation function that converts your pandas DataFrame
    into the format required by GluonTS models.
    
    Args:
        df: DataFrame with time series data (must have 'Date' or 'date' column)
        target_column: Name of the column to forecast (e.g., 'Daily_Cases_MA7')
        freq: Frequency of the time series ('D' for daily, 'H' for hourly, etc.)
        prediction_length: Number of time steps to forecast (used for validation)
        past_feat_columns: Optional list of column names to use as dynamic features
        
    Returns:
        GluonTS ListDataset ready for training/testing
        
    Example:
        >>> df = pd.DataFrame({'Date': [...], 'Daily_Cases_MA7': [...]})
        >>> train_ds = create_gluonts_dataset(
        ...     df=train_df,
        ...     target_column='Daily_Cases_MA7',
        ...     freq='D',
        ...     prediction_length=14
        ... )
    """
    # Ensure we have a date column
    if 'Date' not in df.columns and 'date' not in df.columns:
        raise ValueError("DataFrame must have a 'Date' or 'date' column")
    
    date_col = 'Date' if 'Date' in df.columns else 'date'
    
    # Get the start date
    start_date = pd.to_datetime(df[date_col].iloc[0])
    
    # Get target values
    target = df[target_column].values.tolist()
    
    # Create dataset entry
    data_entry = {
        "start": start_date,
        "target": target
    }
    
    # Add dynamic features if specified (exogenous variables)
    if past_feat_columns:
        feat_dynamic_real = []
        for col in past_feat_columns:
            if col in df.columns:
                feat_dynamic_real.append(df[col].values.tolist())
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")
        
        if feat_dynamic_real:
            data_entry["feat_dynamic_real"] = feat_dynamic_real
    
    # Create ListDataset (single time series)
    dataset = ListDataset(
        [data_entry],
        freq=freq
    )
    
    return dataset


def verify_dataset(dataset: ListDataset, name: str = "Dataset") -> None:
    """
    Verify and print information about a GluonTS dataset.
    
    Useful for debugging and ensuring your dataset is properly formatted.
    
    Args:
        dataset: GluonTS ListDataset to verify
        name: Name to display in output (e.g., "Train", "Test")
        
    Example:
        >>> verify_dataset(train_ds, "Train")
        Dataset: Train
        ✓ Valid GluonTS ListDataset
        - Number of time series: 1
        - Target length: 1000 points
        - Has dynamic features: Yes (8 features)
    """
    try:
        # Convert to list to inspect
        data_list = list(dataset)
        
        print(f"\n{name} Dataset Info:")
        print("=" * 50)
        print(f"✓ Valid GluonTS ListDataset")
        print(f"  Number of time series: {len(data_list)}")
        
        if data_list:
            first_entry = data_list[0]
            print(f"  Start date: {first_entry['start']}")
            print(f"  Target length: {len(first_entry['target'])} points")
            
            if 'feat_dynamic_real' in first_entry:
                n_features = len(first_entry['feat_dynamic_real'])
                print(f"  Dynamic features: Yes ({n_features} features)")
            else:
                print(f"  Dynamic features: No")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")


def prepare_train_test_split(
    full_df: pd.DataFrame,
    test_size: int = 14,
    target_column: str = 'Daily_Cases_MA7'
) -> tuple:
    """
    Split DataFrame into train and test sets for time series.
    
    Args:
        full_df: Complete DataFrame with all data
        test_size: Number of days to reserve for testing
        target_column: Column name to check for NaN values
        
    Returns:
        Tuple of (train_df, test_df)
        
    Example:
        >>> train_df, test_df = prepare_train_test_split(merged_df, test_size=14)
        >>> print(f"Train: {len(train_df)} days, Test: {len(test_df)} days")
    """
    # Remove rows with NaN in target column
    df_clean = full_df.dropna(subset=[target_column]).copy()
    
    # Split into train and test
    split_idx = len(df_clean) - test_size
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()
    
    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(train_df)} days ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  Test:  {len(test_df)} days ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Get list of potential feature columns from DataFrame.
    
    Helper function to identify which columns can be used as exogenous features.
    
    Args:
        df: DataFrame to analyze
        exclude_cols: Columns to exclude (e.g., ['Date', 'Daily_Cases'])
        
    Returns:
        List of column names suitable as features
        
    Example:
        >>> features = get_feature_columns(merged_df, exclude_cols=['Date', 'Daily_Cases_MA7'])
        >>> print(f"Available features: {features}")
    """
    if exclude_cols is None:
        exclude_cols = ['Date', 'date']
    
    # Get all numeric columns except excluded ones
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and 
                   df[col].dtype in ['int64', 'float64']]
    
    return feature_cols


def summary_statistics(df: pd.DataFrame, target_column: str) -> None:
    """
    Print summary statistics for the target variable.
    
    Args:
        df: DataFrame with data
        target_column: Column to summarize
        
    Example:
        >>> summary_statistics(merged_df, 'Daily_Cases_MA7')
    """
    print(f"\n{target_column} Statistics:")
    print("=" * 50)
    print(f"  Count:  {df[target_column].count()}")
    print(f"  Mean:   {df[target_column].mean():.2f}")
    print(f"  Median: {df[target_column].median():.2f}")
    print(f"  Std:    {df[target_column].std():.2f}")
    print(f"  Min:    {df[target_column].min():.2f}")
    print(f"  Max:    {df[target_column].max():.2f}")
    print("=" * 50)


# Quick reference
if __name__ == "__main__":
    print("=" * 70)
    print("GluonTS Data Preparation Utilities")
    print("=" * 70)
    print("\nAvailable functions:")
    print("\n1. create_gluonts_dataset()")
    print("   - Main function: Convert pandas DataFrame → GluonTS ListDataset")
    print("   - Handles date columns, target variables, dynamic features")
    print()
    print("2. verify_dataset()")
    print("   - Check that your dataset is properly formatted")
    print("   - Print dataset information")
    print()
    print("3. prepare_train_test_split()")
    print("   - Split DataFrame into train/test sets")
    print("   - Handles time series ordering")
    print()
    print("4. get_feature_columns()")
    print("   - Helper to identify potential feature columns")
    print()
    print("5. summary_statistics()")
    print("   - Print statistics for target variable")
    print()
    print("=" * 70)
    print("Usage: Import these functions in your notebooks to prepare data")
    print("Model training happens in the notebook files (.API.ipynb)")
    print("=" * 70)
