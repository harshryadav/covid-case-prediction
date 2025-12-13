"""
Data Loader for GluonTS Notebooks

This module provides a simple one-function loader to get COVID-19 data
ready for GluonTS models. This will load US COVID-19 cases, deaths, 
and Google mobility data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path

from GluonTS_utils_data_io import DataLoader
from GluonTS_utils_preprocessing import (
    aggregate_to_national,
    extract_national_mobility,
    merge_all_data
)
from GluonTS_utils_gluonts import (
    create_gluonts_dataset,
    prepare_train_test_split
)


def load_covid_data_for_gluonts(
    data_dir: str = "data",
    target_column: str = "Daily_Cases_MA7",
    test_size: int = 14,
    prediction_length: int = 14,
    use_features: bool = True,
    feature_subset: str = "minimal"
) -> Dict:
    """
    One-stop function to load US COVID-19 data and prepare for GluonTS.
    
    This function handles everything:
    1. Load raw COVID data (cases, deaths, mobility from Google)
    2. Preprocess and aggregate to national level
    3. Merge all sources
    4. Split into train/test
    5. Convert to GluonTS format
    6. Return everything ready to use
    
    Note: Currently uses cases, deaths, and mobility data. Vaccine data
    (data/vaccine.csv) is available for future enhancements.
    
    Args:
        data_dir: Directory containing CSV files
        target_column: Column to forecast (default: 'Daily_Cases_MA7')
        test_size: Number of days for testing (default: 14)
        prediction_length: Forecast horizon (default: 14)
        use_features: Whether to include exogenous features (default: True)
        feature_subset: Which features to use:
            - "minimal": Just deaths (3 features)
            - "moderate": Deaths + key mobility (6 features)
            - "full": All available features (10+ features)
    
    Returns:
        Dictionary with:
            - 'train_ds': GluonTS training dataset
            - 'test_ds': GluonTS testing dataset
            - 'train_df': Training DataFrame (for plotting)
            - 'test_df': Testing DataFrame (for plotting)
            - 'merged_df': Complete merged DataFrame
            - 'target': Name of target column
            - 'features': List of feature columns used
            - 'info': Metadata about the data
    
    Example:
        >>> # Quick load with minimal features
        >>> data = load_covid_data_for_gluonts(feature_subset="minimal")
        >>> train_ds = data['train_ds']
        >>> test_ds = data['test_ds']
        >>> 
        >>> # Train your model
        >>> predictor = estimator.train(train_ds)
    """
    print("=" * 70)
    print("COVID-19 DATA LOADER")
    print("=" * 70)
    
    # Step 1: Load raw data
    print("\n📥 Loading raw data...")
    loader = DataLoader(data_dir=data_dir)
    
    try:
        cases_df = loader.load_cases()
        deaths_df = loader.load_deaths()
        mobility_df = loader.load_mobility()
        # Note: Vaccine data available in data/vaccine.csv for future enhancements
        print("✓ Data files loaded successfully (cases, deaths, mobility)")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print(f"   Make sure data files exist in '{data_dir}/' folder")
        raise
    
    # Step 2: Preprocess
    print("\n🔧 Preprocessing...")
    national_cases = aggregate_to_national(cases_df, data_type='cases')
    national_deaths = aggregate_to_national(deaths_df, data_type='deaths')
    national_mobility = extract_national_mobility(mobility_df)
    
    # Step 3: Merge
    print("\n🔗 Merging data sources...")
    merged_df = merge_all_data(
        national_cases,
        national_deaths,
        national_mobility
    )
    
    print(f"✓ Merged data: {len(merged_df)} days")
    print(f"  Date range: {merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}")
    
    # Step 4: Select features based on subset
    print(f"\n🎯 Feature selection: {feature_subset}")
    
    if not use_features:
        feature_columns = None
        print("  Using target only (no exogenous features)")
    else:
        if feature_subset == "minimal":
            # Just deaths features
            feature_columns = [
                'Daily_Deaths_MA7',
                'Cumulative_Deaths',
                'CFR'
            ]
        elif feature_subset == "moderate":
            # Deaths + key mobility
            feature_columns = [
                'Daily_Deaths_MA7',
                'CFR',
                'retail_and_recreation_percent_change_from_baseline',
                'grocery_and_pharmacy_percent_change_from_baseline',
                'workplaces_percent_change_from_baseline',
                'residential_percent_change_from_baseline'
            ]
        else:  # full
            # All available features except target
            exclude = ['Date', target_column, 'Daily_Cases', 'Cumulative_Cases', 
                      'Daily_Deaths', 'Cumulative_Deaths']
            feature_columns = [col for col in merged_df.columns 
                              if col not in exclude and merged_df[col].dtype in ['int64', 'float64']]
        
        print(f"  Selected {len(feature_columns)} features:")
        for i, feat in enumerate(feature_columns[:5], 1):
            print(f"    {i}. {feat}")
        if len(feature_columns) > 5:
            print(f"    ... and {len(feature_columns) - 5} more")
    
    # Step 5: Split train/test
    print(f"\n✂️  Splitting data (test size: {test_size} days)...")
    train_df, test_df = prepare_train_test_split(
        merged_df,
        test_size=test_size,
        target_column=target_column
    )
    
    # Step 6: Convert to GluonTS format
    print("\n🔄 Converting to GluonTS format...")
    
    # Train dataset: only training period
    train_ds = create_gluonts_dataset(
        df=train_df,
        target_column=target_column,
        freq='D',
        prediction_length=prediction_length,
        past_feat_columns=feature_columns
    )
    
    # Test dataset: FULL DATA (train + test) - GluonTS needs full history for prediction!
    # This is the key fix: test_ds should contain the entire time series
    test_ds = create_gluonts_dataset(
        df=merged_df.dropna(subset=[target_column]),  # Use full merged data, not just test_df
        target_column=target_column,
        freq='D',
        prediction_length=prediction_length,
        past_feat_columns=feature_columns
    )
    
    print("✓ GluonTS datasets created")
    print("  Note: Test dataset contains full time series (train + test periods)")
    
    # Prepare return info
    info = {
        'total_days': len(merged_df),
        'train_days': len(train_df),
        'test_days': len(test_df),
        'date_range': f"{merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}",
        'target_column': target_column,
        'num_features': len(feature_columns) if feature_columns else 0,
        'feature_subset': feature_subset
    }
    
    print("\n" + "=" * 70)
    print("✅ DATA READY FOR TRAINING!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Target: {target_column}")
    print(f"  Features: {info['num_features']} ({feature_subset})")
    print(f"  Train: {info['train_days']} days")
    print(f"  Test: {info['test_days']} days")
    print(f"  Prediction length: {prediction_length} days")
    print("=" * 70)
    
    return {
        'train_ds': train_ds,
        'test_ds': test_ds,
        'train_df': train_df,
        'test_df': test_df,
        'merged_df': merged_df,
        'target': target_column,
        'features': feature_columns,
        'info': info
    }


def quick_load_minimal() -> Dict:
    """
    Quickest load - minimal features, good for testing.
    
    Returns:
        Same as load_covid_data_for_gluonts()
        
    Example:
        >>> data = quick_load_minimal()
        >>> predictor = estimator.train(data['train_ds'])
    """
    return load_covid_data_for_gluonts(feature_subset="minimal")


def quick_load_moderate() -> Dict:
    """
    Moderate features - balanced speed and accuracy.
    
    Returns:
        Same as load_covid_data_for_gluonts()
    """
    return load_covid_data_for_gluonts(feature_subset="moderate")


def quick_load_full() -> Dict:
    """
    All features - maximum information.
    
    Returns:
        Same as load_covid_data_for_gluonts()
    """
    return load_covid_data_for_gluonts(feature_subset="full")


# Quick reference
if __name__ == "__main__":
    print("=" * 70)
    print("Quick Data Loader for Notebooks")
    print("=" * 70)
    print("\nUsage:")
    print("\n1. Full control:")
    print("   from GluonTS_utils_notebook_loader import load_covid_data_for_gluonts")
    print("   data = load_covid_data_for_gluonts(feature_subset='minimal')")
    print("\n2. Quick shortcuts:")
    print("   from GluonTS_utils_notebook_loader import quick_load_minimal")
    print("   data = quick_load_minimal()")
    print("\n3. Access data:")
    print("   train_ds = data['train_ds']")
    print("   test_ds = data['test_ds']")
    print("   train_df = data['train_df']  # For plotting")
    print("\nFeature subsets:")
    print("  • 'minimal':  3 features  (fastest)")
    print("  • 'moderate': 6 features  (balanced)")
    print("  • 'full':     10+ features (most accurate)")
    print("=" * 70)

