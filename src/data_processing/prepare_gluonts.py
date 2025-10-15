"""
Prepare data in GluonTS format for time series forecasting
"""

import pandas as pd
import json
from pathlib import Path
from gluonts.dataset.common import ListDataset


def create_gluonts_dataset(
    df,
    target_column='Daily_MA7',
    start_date=None,
    freq='D',
    prediction_length=14
):
    """
    Convert pandas DataFrame to GluonTS ListDataset
    
    Args:
        df: DataFrame with time series data
        target_column: Column to use as target variable
        start_date: Start date (if None, uses first date in df)
        freq: Frequency string ('D' for daily)
        prediction_length: Forecast horizon
        
    Returns:
        GluonTS ListDataset
    """
    if start_date is None:
        start_date = pd.Timestamp(df['Date'].iloc[0])
    
    # Get target values
    target = df[target_column].fillna(0).values
    
    # Create dataset
    data = [{
        "start": start_date,
        "target": target.tolist()
    }]
    
    return ListDataset(data, freq=freq)


def create_train_test_split(df, test_days=60):
    """
    Split data into train and test sets
    
    Args:
        df: DataFrame with time series data
        test_days: Number of days to reserve for testing
        
    Returns:
        train_df, test_df
    """
    split_idx = len(df) - test_days
    train_df = df.iloc[:split_idx].copy()
    test_df = df.copy()  # Test includes all data for evaluation
    
    return train_df, test_df


def save_metadata(output_dir, train_df, test_df, prediction_length, freq):
    """Save dataset metadata as JSON"""
    metadata = {
        "prediction_length": prediction_length,
        "freq": freq,
        "train_start": str(train_df['Date'].iloc[0]),
        "train_end": str(train_df['Date'].iloc[-1]),
        "train_days": len(train_df),
        "test_start": str(test_df['Date'].iloc[0]),
        "test_end": str(test_df['Date'].iloc[-1]),
        "test_days": len(test_df)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


if __name__ == "__main__":
    # Configuration
    PREDICTION_LENGTH = 14  # Forecast 14 days ahead
    TEST_DAYS = 60  # Reserve last 60 days for testing
    FREQ = 'D'  # Daily frequency
    
    # Load processed data
    data_path = Path('data/processed/national_data.csv')
    if not data_path.exists():
        print("Error: Run preprocess.py first to create national_data.csv")
        exit(1)
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create train/test split
    print(f"\nSplitting data (test_days={TEST_DAYS})")
    train_df, test_df = create_train_test_split(df, test_days=TEST_DAYS)
    
    print(f"Train: {train_df['Date'].iloc[0]} to {train_df['Date'].iloc[-1]} ({len(train_df)} days)")
    print(f"Test:  {test_df['Date'].iloc[0]} to {test_df['Date'].iloc[-1]} ({len(test_df)} days)")
    
    # Create GluonTS datasets
    print("\nCreating GluonTS datasets...")
    train_ds = create_gluonts_dataset(
        train_df,
        target_column='Daily_MA7',
        freq=FREQ,
        prediction_length=PREDICTION_LENGTH
    )
    
    test_ds = create_gluonts_dataset(
        test_df,
        target_column='Daily_MA7',
        freq=FREQ,
        prediction_length=PREDICTION_LENGTH
    )
    
    # Save datasets
    output_dir = Path('data/gluonts')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metadata
    metadata = save_metadata(output_dir, train_df, test_df, PREDICTION_LENGTH, FREQ)
    
    print(f"\n✓ GluonTS datasets ready!")
    print(f"  Output directory: {output_dir}")
    print(f"  Prediction length: {PREDICTION_LENGTH} days")
    print(f"  Frequency: {FREQ}")
    
    print("\nNext steps:")
    print("  1. Train a model: python src/models/train_baseline.py")
    print("  2. Or jump to: python src/models/train_deepar.py")

