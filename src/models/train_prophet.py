"""
Train Prophet model for COVID-19 forecasting

Prophet is a procedure for forecasting time series data developed by Facebook.
It is robust to missing data and handles outliers well.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Prophet imports
try:
    from prophet import Prophet
    BACKEND = "Prophet"
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install prophet")
    exit(1)


def load_data(test_days=60):
    """Load and split data"""
    # Load processed CSV
    df = pd.read_csv('data/processed/national_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Prepare for Prophet (needs 'ds' and 'y' columns)
    prophet_df = df[['Date', 'Daily_MA7']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['y'] = prophet_df['y'].fillna(0)
    
    # Split into train and test
    train_df = prophet_df.iloc[:-test_days].copy()
    test_df = prophet_df.copy()
    
    return train_df, test_df


def compute_metrics(actual, forecast):
    """Compute evaluation metrics"""
    # Filter to forecast period only
    forecast_period = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Merge with actual
    comparison = actual.merge(forecast_period, on='ds', how='inner')
    comparison = comparison.dropna()
    
    if len(comparison) == 0:
        return None
    
    y_true = comparison['y'].values
    y_pred = comparison['yhat'].values
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100  # +1 to avoid division by zero
    
    # Coverage (how many actuals fall within prediction intervals)
    within_50 = np.mean((y_true >= comparison['yhat_lower']) & (y_true <= comparison['yhat_upper']))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'abs_error': mae,  # MAE
        'MAPE': mape / 100,  # As fraction
        'sMAPE': mape / 100,  # Simplified
        'mean_wQuantileLoss': mae / np.mean(y_true),  # Normalized MAE as proxy
        'Coverage[0.5]': within_50,
    }


if __name__ == "__main__":
    print("="*60)
    print("PROPHET MODEL TRAINING")
    print("="*60)
    
    # Check if data is prepared
    data_file = Path('data/processed/national_data.csv')
    if not data_file.exists():
        print("\nError: Processed data not found!")
        print("Run: python src/data_processing/preprocess.py")
        exit(1)
    
    # Configuration
    PREDICTION_LENGTH = 14
    TEST_DAYS = 60
    
    print("\nConfiguration:")
    print(f"  Prediction length: {PREDICTION_LENGTH} days")
    print(f"  Model: Prophet (Facebook)")
    
    # Load data
    print("\n[1/4] Loading data...")
    train_df, test_df = load_data(test_days=TEST_DAYS)
    print(f"✓ Data loaded: {len(train_df)} training days")
    
    # Initialize and train model
    print(f"\n[2/4] Training Prophet model...")
    model = Prophet(
        growth='linear',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode='additive',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    
    model.fit(train_df)
    print("✓ Model trained")
    
    # Make forecast
    print("\n[3/4] Generating forecasts...")
    future = model.make_future_dataframe(periods=PREDICTION_LENGTH, freq='D')
    forecast = model.predict(future)
    print("✓ Forecast generated")
    
    # Evaluate on test period
    print("\n[4/4] Evaluating model...")
    test_period = test_df.iloc[-PREDICTION_LENGTH:]
    metrics = compute_metrics(test_period, forecast)
    
    if metrics:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'MAPE' in key or 'sMAPE' in key:
                    print(f"{key:20s}: {value:.2%}")
                elif 'Coverage' in key:
                    print(f"{key:20s}: {value:.2f}")
                else:
                    print(f"{key:20s}: {value:.2f}")
    
    # Plot
    print("\n[5/5] Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot last 90 days of actual data
    plot_df = test_df.iloc[-90:].copy()
    ax.plot(plot_df['ds'], plot_df['y'], 'k-', linewidth=2, label='Actual')
    
    # Plot forecast
    forecast_plot = forecast[forecast['ds'].isin(test_period['ds'])]
    ax.plot(forecast_plot['ds'], forecast_plot['yhat'], 'C2-', linewidth=2, label='Forecast')
    ax.fill_between(forecast_plot['ds'], 
                     forecast_plot['yhat_lower'], 
                     forecast_plot['yhat_upper'],
                     alpha=0.3, color='C2', label='80% CI')
    
    ax.set_title('COVID-19 Case Forecast (Prophet Model)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Daily Cases (7-day MA)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'prophet_forecast.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_dir / 'prophet_forecast.png'}")
    
    # Save metrics
    if metrics:
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_dir / 'prophet_metrics.csv', index=False)
        print(f"✓ Metrics saved to {output_dir / 'prophet_metrics.csv'}")
    
    print("\n" + "="*60)
    print("PROPHET TRAINING COMPLETE! 🎉")
    print("="*60)

