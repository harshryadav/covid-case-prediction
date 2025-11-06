"""
Train WaveNet model for COVID-19 forecasting

WaveNet uses dilated causal convolutions to capture temporal patterns
with a large receptive field efficiently.
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
    from gluonts.torch.model.wavenet import WaveNetEstimator
    from gluonts.evaluation import make_evaluation_predictions, Evaluator
    import torch
    BACKEND = "PyTorch"
    
    # Fix for macOS MPS device compatibility
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if torch.backends.mps.is_available():
        print("Using CPU (MPS fallback enabled)")
        DEVICE = "cpu"
    else:
        DEVICE = "cpu"
        
except ImportError:
    print("Error: GluonTS with PyTorch not installed.")
    print("Install with: pip install gluonts torch lightning")
    exit(1)


# Configuration
PREDICTION_LENGTH = 14  # 2-week forecast horizon
CONTEXT_LENGTH = 56     # 8 weeks of history
EPOCHS = 10
NUM_RESIDUAL_CHANNELS = 24
NUM_SKIP_CHANNELS = 32
DILATION_DEPTH = 4
NUM_STACKS = 3


def load_gluonts_data(data_dir='data/gluonts'):
    """Load processed data for training"""
    data_dir = Path(data_dir)
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load processed CSV
    df = pd.read_csv('data/processed/national_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    start_date = pd.Timestamp(metadata['train_start'])
    freq = metadata['freq']
    
    # Split data based on train_days
    train_end_idx = metadata['train_days']
    train_df = df.iloc[:train_end_idx]
    
    # Create datasets
    train_target = train_df['Daily_MA7'].fillna(0).values
    train_data = [{"start": start_date, "target": train_target.tolist()}]
    train_ds = ListDataset(train_data, freq=freq)
    
    test_target = df['Daily_MA7'].fillna(0).values
    test_data = [{"start": start_date, "target": test_target.tolist()}]
    test_ds = ListDataset(test_data, freq=freq)
    
    return train_ds, test_ds, metadata, df


def train_wavenet(train_ds, metadata):
    """Train WaveNet model"""
    print("\n" + "="*60)
    print("WAVENET MODEL TRAINING")
    print("="*60)
    print("\nConfiguration:")
    print(f"  Prediction length: {PREDICTION_LENGTH} days")
    print(f"  Context length: {CONTEXT_LENGTH} days")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Architecture: Dilated CNN (WaveNet)")
    print(f"  Residual channels: {NUM_RESIDUAL_CHANNELS}")
    print(f"  Skip channels: {NUM_SKIP_CHANNELS}")
    print(f"  Dilation depth: {DILATION_DEPTH}")
    print(f"  Num stacks: {NUM_STACKS}")
    
    print("\n[1/4] Initializing WaveNet model...")
    
    estimator = WaveNetEstimator(
        freq="D",
        prediction_length=PREDICTION_LENGTH,
        num_residual_channels=NUM_RESIDUAL_CHANNELS,
        num_skip_channels=NUM_SKIP_CHANNELS,
        dilation_depth=DILATION_DEPTH,
        num_stacks=NUM_STACKS,
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
    
    print("\n[2/4] Training model...")
    print("(This may take 5-10 minutes...)")
    
    predictor = estimator.train(train_ds)
    
    print("\n✓ Training complete!")
    
    return predictor


def evaluate_model(predictor, test_ds, metadata):
    """Evaluate model performance"""
    print("\n[3/4] Evaluating model...")
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Print key metrics in clean format
    for key, value in agg_metrics.items():
        if isinstance(value, (int, float)):
            if 'MAPE' in key or 'sMAPE' in key or 'MASE' in key:
                print(f"{key:30s}: {value:.2%}")
            else:
                print(f"{key:30s}: {value:.2f}")
    
    return forecasts, tss, agg_metrics


def create_visualization(forecasts, tss, metadata, df):
    """Create clean, professional forecast visualization"""
    print("\n[4/4] Creating visualization...")
    
    forecast = forecasts[0]
    ts = tss[0]
    
    # Get test period data (last 60 days for actual plotting, 14 for forecast)
    test_start_idx = metadata['train_days']
    test_df = df.iloc[test_start_idx:test_start_idx + 60].copy()
    actual_values = test_df['Daily_MA7'].values
    
    # Get forecast statistics
    forecast_median = forecast.quantile(0.5)
    forecast_lower = forecast.quantile(0.1)
    forecast_upper = forecast.quantile(0.9)
    
    # Create figure with clean styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual values (full test period)
    dates = test_df['Date'].values
    ax.plot(dates, actual_values, 
            'o-', color='#2E86AB', linewidth=2.5, 
            markersize=6, label='Actual Cases (7-day MA)',
            zorder=3)
    
    # Plot forecast (last 14 days)
    forecast_dates = dates[-PREDICTION_LENGTH:]
    ax.plot(forecast_dates, forecast_median,
            's-', color='#A23B72', linewidth=2.5,
            markersize=7, label='WaveNet Forecast (Median)',
            zorder=2)
    
    # Plot prediction intervals with clean fill
    ax.fill_between(forecast_dates,
                     forecast_lower, forecast_upper,
                     alpha=0.25, color='#A23B72',
                     label='90% Prediction Interval',
                     zorder=1)
    
    # Styling
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Daily New Cases (7-day Moving Average)', 
                  fontsize=13, fontweight='bold')
    ax.set_title('WaveNet COVID-19 Forecast - 14-Day Prediction',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Clean legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
              edgecolor='gray', fancybox=True)
    
    # Format x-axis for better date display
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics box
    mae = np.mean(np.abs(actual_values[-PREDICTION_LENGTH:] - forecast_median))
    textstr = f'Forecast Period: {PREDICTION_LENGTH} days\n'
    textstr += f'MAE: {mae:,.0f} cases/day'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'wavenet_forecast.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved to {plot_path}")
    
    return plot_path


def save_metrics(agg_metrics):
    """Save metrics to CSV"""
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    metrics_df = pd.DataFrame([agg_metrics])
    metrics_path = output_dir / 'wavenet_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    
    print(f"✓ Metrics saved to {metrics_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WAVENET TRAINING PIPELINE")
    print("="*60)
    print(f"\nBackend: {BACKEND}")
    print(f"Device: {DEVICE}")
    
    # Load data
    print("\nLoading data...")
    train_ds, test_ds, metadata, df = load_gluonts_data()
    print("✓ Data loaded")
    
    # Train model
    predictor = train_wavenet(train_ds, metadata)
    
    # Evaluate
    forecasts, tss, agg_metrics = evaluate_model(predictor, test_ds, metadata)
    
    # Visualize
    plot_path = create_visualization(forecasts, tss, metadata, df)
    
    # Save metrics
    save_metrics(agg_metrics)
    
    print("\n" + "="*60)
    print("WAVENET TRAINING COMPLETE! 🎉")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  📊 Plot: {plot_path}")
    print(f"  📈 Metrics: results/wavenet_metrics.csv")
    print("\n" + "="*60)

