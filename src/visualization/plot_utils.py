"""
Plotting utilities for COVID-19 forecasting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_time_series(df, date_col='Date', value_col='Daily_MA7', title='COVID-19 Cases'):
    """
    Plot a simple time series
    
    Args:
        df: DataFrame with time series data
        date_col: Name of date column
        value_col: Name of value column
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df[date_col], df[value_col], linewidth=2, color='steelblue')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Cases (7-day MA)')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_forecast_comparison(actual, forecasts_dict, dates):
    """
    Plot multiple model forecasts against actual values
    
    Args:
        actual: Actual values (numpy array)
        forecasts_dict: Dict of model_name -> predictions
        dates: Date array
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual
    ax.plot(dates, actual, 'o-', label='Actual', linewidth=2, markersize=4)
    
    # Plot forecasts
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, forecast) in enumerate(forecasts_dict.items()):
        ax.plot(dates, forecast, 's--', label=model_name, 
                alpha=0.7, color=colors[i % len(colors)])
    
    ax.set_title('Forecast Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Cases (7-day MA)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_metrics_comparison(metrics_df):
    """
    Plot bar chart comparing model metrics
    
    Args:
        metrics_df: DataFrame with columns [Model, MAE, RMSE, MAPE]
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['MAE', 'RMSE', 'MAPE']
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for ax, metric, color in zip(axes, metrics, colors):
        ax.bar(metrics_df['Model'], metrics_df[metric], color=color, alpha=0.7)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_with_uncertainty(dates, actual, median, lower, upper, title='Forecast with Uncertainty'):
    """
    Plot forecast with confidence intervals
    
    Args:
        dates: Date array
        actual: Actual values
        median: Median forecast
        lower: Lower confidence bound
        upper: Upper confidence bound
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual
    ax.plot(dates, actual, 'o-', label='Actual', color='black', linewidth=2)
    
    # Plot median forecast
    ax.plot(dates, median, 's-', label='Forecast (median)', color='blue', linewidth=2)
    
    # Plot confidence interval
    ax.fill_between(dates, lower, upper, alpha=0.3, color='blue', label='90% CI')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Cases (7-day MA)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def save_figure(fig, filename, dpi=150):
    """Save figure to results directory"""
    from pathlib import Path
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved plot to {filepath}")
    
    return filepath


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    actual = np.random.randn(100).cumsum() + 100
    forecast = actual + np.random.randn(100) * 5
    
    # Plot
    fig, ax = plot_time_series(
        pd.DataFrame({'Date': dates, 'Daily_MA7': actual}),
        title='Sample Time Series'
    )
    
    save_figure(fig, 'sample_plot.png')
    print("Example plot created!")

