"""
Train baseline forecasting models (Naive, Seasonal Naive)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class NaiveForecaster:
    """Naive forecasting: repeat the last observed value"""
    
    def __init__(self):
        self.last_value = None
    
    def fit(self, y):
        """Store the last value"""
        self.last_value = y[-1]
    
    def predict(self, horizon):
        """Predict by repeating last value"""
        return np.repeat(self.last_value, horizon)


class SeasonalNaiveForecaster:
    """Seasonal Naive: repeat values from same day last week"""
    
    def __init__(self, season_length=7):
        self.season_length = season_length
        self.history = None
    
    def fit(self, y):
        """Store the historical values"""
        self.history = y
    
    def predict(self, horizon):
        """Predict using seasonal pattern"""
        predictions = []
        for h in range(horizon):
            # Use value from season_length days ago
            idx = len(self.history) - self.season_length + (h % self.season_length)
            predictions.append(self.history[idx])
        return np.array(predictions)


def evaluate_forecast(actual, predicted):
    """Calculate forecast metrics"""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


if __name__ == "__main__":
    # Load processed data
    data_path = Path('data/processed/national_data.csv')
    if not data_path.exists():
        print("Error: Run preprocess.py first!")
        exit(1)
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use 7-day moving average as target
    y = df['Daily_MA7'].fillna(0).values
    dates = df['Date'].values
    
    # Train/test split (last 30 days for testing)
    test_days = 30
    train_y = y[:-test_days]
    test_y = y[-test_days:]
    test_dates = dates[-test_days:]
    
    print("="*60)
    print("BASELINE FORECASTING MODELS")
    print("="*60)
    print(f"\nTraining data: {len(train_y)} days")
    print(f"Test data: {len(test_y)} days")
    
    # Naive Forecaster
    print("\n[1/2] Training Naive Forecaster...")
    naive = NaiveForecaster()
    naive.fit(train_y)
    naive_pred = naive.predict(test_days)
    naive_metrics = evaluate_forecast(test_y, naive_pred)
    
    print(f"  MAE:  {naive_metrics['MAE']:.2f}")
    print(f"  RMSE: {naive_metrics['RMSE']:.2f}")
    print(f"  MAPE: {naive_metrics['MAPE']:.2f}%")
    
    # Seasonal Naive Forecaster
    print("\n[2/2] Training Seasonal Naive Forecaster (7-day)...")
    seasonal_naive = SeasonalNaiveForecaster(season_length=7)
    seasonal_naive.fit(train_y)
    seasonal_pred = seasonal_naive.predict(test_days)
    seasonal_metrics = evaluate_forecast(test_y, seasonal_pred)
    
    print(f"  MAE:  {seasonal_metrics['MAE']:.2f}")
    print(f"  RMSE: {seasonal_metrics['RMSE']:.2f}")
    print(f"  MAPE: {seasonal_metrics['MAPE']:.2f}%")
    
    # Plot results
    print("\nCreating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Naive
    ax1.plot(test_dates, test_y, 'o-', label='Actual', linewidth=2)
    ax1.plot(test_dates, naive_pred, 's--', label='Naive Forecast', alpha=0.7)
    ax1.set_title('Naive Forecaster', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Daily Cases (7-day MA)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Seasonal Naive
    ax2.plot(test_dates, test_y, 'o-', label='Actual', linewidth=2)
    ax2.plot(test_dates, seasonal_pred, 's--', label='Seasonal Naive Forecast', alpha=0.7)
    ax2.set_title('Seasonal Naive Forecaster (7-day)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Cases (7-day MA)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'baseline_forecasts.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_dir / 'baseline_forecasts.png'}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Model': ['Naive', 'Seasonal Naive'],
        'MAE': [naive_metrics['MAE'], seasonal_metrics['MAE']],
        'RMSE': [naive_metrics['RMSE'], seasonal_metrics['RMSE']],
        'MAPE': [naive_metrics['MAPE'], seasonal_metrics['MAPE']]
    })
    
    metrics_df.to_csv(output_dir / 'baseline_metrics.csv', index=False)
    print(f"✓ Saved metrics to {output_dir / 'baseline_metrics.csv'}")
    
    print("\n" + "="*60)
    print("BASELINE MODELS COMPLETE!")
    print("="*60)
    print("\nNext step: Train DeepAR model")
    print("  python src/models/train_deepar.py")

