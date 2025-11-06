"""
Train DeepVAR model for COVID-19 forecasting

DeepVAR is a multivariate extension of DeepAR that models dependencies
between multiple time series (cases + mobility features).
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# GluonTS imports (PyTorch backend)
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.torch.model.deepvar import DeepVAREstimator
    from gluonts.evaluation import make_evaluation_predictions, Evaluator
    import torch
    BACKEND = "PyTorch"
    
    # Fix for macOS MPS device compatibility
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if torch.backends.mps.is_available():
        print("Note: Using CPU (MPS fallback enabled for compatibility)")
        DEVICE = "cpu"
    else:
        DEVICE = "cpu"
        
except ImportError:
    print("Error: GluonTS with PyTorch not installed.")
    print("Run: pip install gluonts torch lightning")
    exit(1)


def load_gluonts_data(data_dir='data/gluonts'):
    """Load processed data for training"""
    data_dir = Path(data_dir)
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load processed CSV
    df = pd.read_csv('data/processed/national_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df, metadata


def create_multivariate_datasets(df, metadata, test_days=60):
    """Create multivariate train and test datasets"""
    freq = metadata['freq']
    
    # Define mobility columns
    mobility_cols = [
        'retail and recreation', 'grocery and pharmacy',
        'parks', 'transit stations', 'workplaces', 'residential'
    ]
    
    # Combine target with mobility features
    target_cols = ['Daily_MA7'] + [col for col in mobility_cols if col in df.columns]
    
    print(f"  Multivariate series: {len(target_cols)} variables")
    print(f"  Variables: {target_cols}")
    
    # Train data (exclude test period)
    train_df = df.iloc[:-test_days].copy()
    
    # Fill NaN values (especially for early dates without mobility data)
    train_data = train_df[target_cols].fillna(method='ffill').fillna(0)
    
    # Create 2D array (num_series x time_steps)
    train_target = train_data.values.T.tolist()
    
    train_ds = ListDataset(
        [{
            "start": pd.Timestamp(train_df['Date'].iloc[0]),
            "target": train_target
        }],
        freq=freq
    )
    
    # Test data (full dataset)
    test_data = df[target_cols].fillna(method='ffill').fillna(0)
    test_target = test_data.values.T.tolist()
    
    test_ds = ListDataset(
        [{
            "start": pd.Timestamp(df['Date'].iloc[0]),
            "target": test_target
        }],
        freq=freq
    )
    
    return train_ds, test_ds, len(target_cols)


if __name__ == "__main__":
    print("="*60)
    print("DEEPVAR MODEL TRAINING (Multivariate)")
    print("="*60)
    
    # Check if data is prepared
    data_dir = Path('data/gluonts')
    if not data_dir.exists():
        print("\nError: GluonTS data not prepared!")
        print("Run: python src/data_processing/prepare_gluonts.py")
        exit(1)
    
    # Configuration
    PREDICTION_LENGTH = 14
    CONTEXT_LENGTH = 56  # Use 8 weeks of history
    EPOCHS = 10
    
    print("\nConfiguration:")
    print(f"  Prediction length: {PREDICTION_LENGTH} days")
    print(f"  Context length: {CONTEXT_LENGTH} days")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Model: DeepVAR (Multivariate)")
    
    # Load data
    print("\n[1/4] Loading data...")
    df, metadata = load_gluonts_data()
    train_ds, test_ds, num_series = create_multivariate_datasets(df, metadata, test_days=60)
    print("✓ Data loaded")
    
    # Initialize model
    print(f"\n[2/4] Initializing DeepVAR model ({BACKEND} backend on {DEVICE.upper()})...")
    estimator = DeepVAREstimator(
        freq="D",
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        target_dim=num_series,  # Number of time series (7: cases + 6 mobility)
        hidden_dim=40,
        num_layers=2,
        dropout_rate=0.1,
        lr=1e-3,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={
            "max_epochs": EPOCHS,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "accelerator": DEVICE
        }
    )
    print("✓ Model initialized")
    
    # Train
    print("\n[3/4] Training model...")
    print("(This may take 5-15 minutes due to multivariate modeling...)")
    predictor = estimator.train(training_data=train_ds)
    print("✓ Training complete!")
    
    # Evaluate
    print("\n[4/4] Evaluating model...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    # Note: For multivariate, we evaluate the first dimension (cases)
    # Extract first dimension from multivariate forecasts
    univariate_forecasts = []
    univariate_tss = []
    
    for forecast, ts in zip(forecasts, tss):
        # Get first dimension (Daily_MA7 cases)
        univariate_forecasts.append(forecast.copy_dim(0))
        univariate_tss.append(ts[0])
    
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(univariate_tss, univariate_forecasts)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Cases Dimension)")
    print("="*60)
    
    # Display metrics
    for key, value in agg_metrics.items():
        if isinstance(value, (int, float)):
            if 'MAPE' in key or 'sMAPE' in key or 'MASE' in key:
                print(f"{key:20s}: {value:.2%}")
            else:
                print(f"{key:20s}: {value:.2f}")
    
    # Plot
    print("\n[5/5] Creating visualization...")
    plt.figure(figsize=(14, 6))
    
    ts = univariate_tss[0]
    forecast = univariate_forecasts[0]
    
    # Plot last 90 days of actual data
    ts[-90:].plot(label='Actual', color='black', linewidth=2)
    
    # Plot forecast with confidence intervals
    forecast.plot(color='C3')
    
    plt.title('COVID-19 Case Forecast (DeepVAR Model - Multivariate)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Daily Cases (7-day MA)')
    plt.xlabel('Date')
    plt.legend(['Actual', 'Forecast (median)', '90% CI', '50% CI'])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'deepvar_forecast.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_dir / 'deepvar_forecast.png'}")
    
    # Save metrics
    metrics_df = pd.DataFrame([agg_metrics])
    metrics_df.to_csv(output_dir / 'deepvar_metrics.csv', index=False)
    print(f"✓ Metrics saved to {output_dir / 'deepvar_metrics.csv'}")
    
    print("\n" + "="*60)
    print("DEEPVAR TRAINING COMPLETE! 🎉")
    print("="*60)
    print("\nNote: DeepVAR models dependencies between:")
    print("  • COVID-19 cases")
    print("  • 6 mobility indicators")
    print("This captures how mobility changes affect case trends!")

