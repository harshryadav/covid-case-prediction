"""
Train DeepAR model for COVID-19 forecasting
"""

import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# GluonTS imports (PyTorch backend)
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.torch.model.deepar import DeepAREstimator
    from gluonts.evaluation import make_evaluation_predictions, Evaluator
    import torch
    BACKEND = "PyTorch"
    
    # Fix for macOS MPS device compatibility
    # Use CPU for now as MPS doesn't support all operations
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


def create_datasets(df, metadata, test_days=60):
    """Create train and test datasets"""
    freq = metadata['freq']
    
    # Train data (exclude test period)
    train_df = df.iloc[:-test_days].copy()
    train_ds = ListDataset(
        [{
            "start": pd.Timestamp(train_df['Date'].iloc[0]),
            "target": train_df['Daily_MA7'].fillna(0).tolist()
        }],
        freq=freq
    )
    
    # Test data (full dataset for evaluation)
    test_ds = ListDataset(
        [{
            "start": pd.Timestamp(df['Date'].iloc[0]),
            "target": df['Daily_MA7'].fillna(0).tolist()
        }],
        freq=freq
    )
    
    return train_ds, test_ds


if __name__ == "__main__":
    print("="*60)
    print("DEEPAR MODEL TRAINING")
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
    EPOCHS = 10  # Start small for testing
    
    print("\nConfiguration:")
    print(f"  Prediction length: {PREDICTION_LENGTH} days")
    print(f"  Context length: {CONTEXT_LENGTH} days")
    print(f"  Epochs: {EPOCHS}")
    
    # Load data
    print("\n[1/4] Loading data...")
    df, metadata = load_gluonts_data()
    train_ds, test_ds = create_datasets(df, metadata, test_days=60)
    print("✓ Data loaded")
    
    # Initialize model (PyTorch backend)
    print(f"\n[2/4] Initializing DeepAR model ({BACKEND} backend on {DEVICE.upper()})...")
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        hidden_size=40,
        num_layers=2,
        dropout_rate=0.1,
        lr=1e-3,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={
            "max_epochs": EPOCHS,
            "enable_progress_bar": True,
            "enable_model_summary": False,  # Reduce output verbosity
            "accelerator": DEVICE
        }
    )
    print("✓ Model initialized")
    
    # Train
    print("\n[3/4] Training model...")
    print("(This may take 5-10 minutes...)")
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
    
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Display available metrics (different naming in PyTorch backend)
    for key, value in agg_metrics.items():
        if isinstance(value, (int, float)):
            if 'MAPE' in key or 'sMAPE' in key or 'MASE' in key:
                print(f"{key:20s}: {value:.2%}")
            else:
                print(f"{key:20s}: {value:.2f}")
    
    # Plot
    print("\n[5/5] Creating visualization...")
    plt.figure(figsize=(14, 6))
    
    ts = tss[0]
    forecast = forecasts[0]
    
    # Plot last 90 days of actual data
    ts[-90:].plot(label='Actual', color='black', linewidth=2)
    
    # Plot forecast with confidence intervals
    forecast.plot(color='C0')
    
    plt.title('COVID-19 Case Forecast (DeepAR Model)', fontsize=14, fontweight='bold')
    plt.ylabel('Daily Cases (7-day MA)')
    plt.xlabel('Date')
    plt.legend(['Actual', 'Forecast (median)', '90% CI', '50% CI'])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'deepar_forecast.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_dir / 'deepar_forecast.png'}")
    
    # Save metrics
    metrics_df = pd.DataFrame([agg_metrics])
    metrics_df.to_csv(output_dir / 'deepar_metrics.csv', index=False)
    print(f"✓ Metrics saved to {output_dir / 'deepar_metrics.csv'}")
    
    print("\n" + "="*60)
    print("DEEPAR TRAINING COMPLETE! 🎉")
    print("="*60)
    
    print("\nNext steps:")
    print("  1. View results: open results/deepar_forecast.png")
    print("  2. Tune hyperparameters (epochs, layers, cells)")
    print("  3. Try Transformer model: train_transformer.py")

