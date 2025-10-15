"""
Train DeepAR model for COVID-19 forecasting
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# GluonTS imports
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.trainer import Trainer
    from gluonts.evaluation import make_evaluation_predictions, Evaluator
except ImportError:
    print("Error: GluonTS not installed. Run: pip install gluonts mxnet")
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
    
    # Initialize model
    print("\n[2/4] Initializing DeepAR model...")
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        num_layers=2,
        num_cells=40,
        dropout_rate=0.1,
        trainer=Trainer(
            epochs=EPOCHS,
            learning_rate=1e-3,
            batch_size=32,
            num_batches_per_epoch=50
        )
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
    print(f"RMSE:  {agg_metrics['RMSE']:.2f}")
    print(f"MAE:   {agg_metrics['MAE']:.2f}")
    print(f"MAPE:  {agg_metrics['MAPE']:.2%}")
    print(f"sMAPE: {agg_metrics['sMAPE']:.2%}")
    
    # Plot
    print("\n[5/5] Creating visualization...")
    plt.figure(figsize=(14, 6))
    
    ts = tss[0]
    forecast = forecasts[0]
    
    # Plot last 90 days of actual data
    ts[-90:].plot(label='Actual', color='black', linewidth=2)
    
    # Plot forecast
    forecast.plot(prediction_intervals=[50, 90], color='blue')
    
    plt.title('COVID-19 Case Forecast (DeepAR Model)', fontsize=14, fontweight='bold')
    plt.ylabel('Daily Cases (7-day MA)')
    plt.xlabel('Date')
    plt.legend(['Actual', 'Forecast (median)', '50% CI', '90% CI'])
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

