"""
Evaluation Metrics Utilities

This module provides common evaluation metrics for time series forecasting.
Use these functions across all model notebooks to avoid code duplication.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def calculate_metrics(
    forecast_values: Union[np.ndarray, pd.Series, list], 
    actual_values: Union[np.ndarray, pd.Series, list]
) -> Dict[str, float]:
    """
    Calculate comprehensive forecasting metrics.
    
    Args:
        forecast_values: Forecasted values (array, Series, or list)
        actual_values: Actual values (array, Series, or list)
        
    Returns:
        Dictionary with metrics: mae, rmse, mape, me, max_error
        
    Example:
        >>> metrics = calculate_metrics(forecast.mean, actual)
        >>> print(f"MAE: {metrics['mae']:.2f}")
    """
    # Convert to numpy arrays to ensure compatibility
    forecast_values = np.asarray(forecast_values).flatten()
    actual_values = np.asarray(actual_values).flatten()
    
    # Ensure same length
    if len(forecast_values) != len(actual_values):
        min_len = min(len(forecast_values), len(actual_values))
        forecast_values = forecast_values[:min_len]
        actual_values = actual_values[:min_len]
    
    errors = forecast_values - actual_values
    
    # Mean Absolute Error
    mae = np.mean(np.abs(errors))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean(errors**2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs(errors / actual_values)) * 100
    
    # Mean Error (bias)
    me = np.mean(errors)
    
    # Maximum Error
    max_error = np.max(np.abs(errors))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'me': me,
        'max_error': max_error
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        model_name: Name of the model for display
        
    Example:
        >>> metrics = calculate_metrics(forecast, actual)
        >>> print_metrics(metrics, "DeepAR")
    """
    print(f"\n📊 {model_name} Performance:")
    print("=" * 60)
    print(f"MAE (Mean Absolute Error):      {metrics['mae']:>10,.2f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:>10,.2f}")
    print(f"MAPE (Mean Abs. % Error):       {metrics['mape']:>10.2f} %")
    print(f"ME (Mean Error / Bias):         {metrics['me']:>10,.2f}")
    print(f"Maximum Error:                   {metrics['max_error']:>10,.2f}")
    print("=" * 60)
    
    # Interpretation
    if metrics['mape'] < 10:
        print("\n✓ Excellent! Error < 10%")
    elif metrics['mape'] < 20:
        print("\n✓ Good performance! Error < 20%")
    else:
        print("\n⚠️  Moderate performance (COVID data is highly variable)")
    
    if abs(metrics['me']) < metrics['mae'] / 2:
        print("✓ Low bias (model not systematically over/under-predicting)")
    else:
        bias_direction = "over" if metrics['me'] > 0 else "under"
        print(f"⚠️  Model tends to {bias_direction}-predict")


def plot_forecast(
    train_df: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    forecast_quantiles: Dict[float, np.ndarray],
    target_column: str,
    model_name: str,
    save_path: str = None,
    context_days: int = 60
) -> None:
    """
    Create a comprehensive forecast visualization.
    
    Args:
        train_df: Training DataFrame with Date and target columns
        forecast_dates: Dates for forecast period
        forecast_values: Mean forecast values
        actual_values: Actual values for forecast period
        forecast_quantiles: Dict of quantiles (e.g., {0.1: array, 0.9: array})
        target_column: Name of target column
        model_name: Model name for title
        save_path: Path to save plot (optional)
        context_days: Number of historical days to show
        
    Example:
        >>> plot_forecast(
        ...     train_df, forecast_dates, forecast.mean, actual,
        ...     {0.1: forecast.quantile(0.1), 0.9: forecast.quantile(0.9)},
        ...     'Daily_Cases_MA7', 'DeepAR', 'forecast.png'
        ... )
    """
    plt.figure(figsize=(16, 6))
    
    # Historical context
    train_context = train_df.tail(context_days)
    plt.plot(train_context['Date'], train_context[target_column],
             label='Historical Data', color='steelblue', linewidth=2, alpha=0.8)
    
    # Actual future values
    plt.plot(forecast_dates, actual_values,
             label='Actual', color='orange', linewidth=3, 
             marker='o', markersize=8, zorder=5)
    
    # Forecast
    plt.plot(forecast_dates, forecast_values,
             label=f'{model_name} Forecast', color='red', linewidth=3,
             marker='s', markersize=7, linestyle='--', zorder=4)
    
    # Confidence intervals
    if 0.05 in forecast_quantiles and 0.95 in forecast_quantiles:
        plt.fill_between(
            forecast_dates,
            forecast_quantiles[0.05],
            forecast_quantiles[0.95],
            alpha=0.15, color='red', label='90% Confidence'
        )
    
    if 0.25 in forecast_quantiles and 0.75 in forecast_quantiles:
        plt.fill_between(
            forecast_dates,
            forecast_quantiles[0.25],
            forecast_quantiles[0.75],
            alpha=0.25, color='red', label='50% Confidence'
        )
    
    plt.title(f'{model_name} Forecast Visualization', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=13)
    plt.ylabel(target_column.replace('_', ' '), fontsize=13)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved as '{save_path}'")
    
    plt.show()


def plot_error_analysis(
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    forecast_quantiles: Dict[float, np.ndarray],
    model_name: str,
    save_path: str = None
) -> None:
    """
    Create detailed error analysis plots.
    
    Args:
        forecast_values: Mean forecast values
        actual_values: Actual values
        forecast_quantiles: Dict of quantiles
        model_name: Model name for title
        save_path: Path to save plot (optional)
        
    Example:
        >>> plot_error_analysis(
        ...     forecast.mean, actual,
        ...     {0.1: forecast.quantile(0.1), 0.9: forecast.quantile(0.9)},
        ...     'DeepAR', 'error_analysis.png'
        ... )
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    forecast_period = len(forecast_values)
    errors = forecast_values - actual_values
    
    # Plot 1: Forecast vs Actual
    axes[0, 0].plot(range(1, forecast_period + 1), actual_values, 'o-', 
                    label='Actual', color='orange', linewidth=2, markersize=8)
    axes[0, 0].plot(range(1, forecast_period + 1), forecast_values, 's--',
                    label='Forecast', color='red', linewidth=2, markersize=7)
    if 0.1 in forecast_quantiles and 0.9 in forecast_quantiles:
        axes[0, 0].fill_between(range(1, forecast_period + 1), 
                                 forecast_quantiles[0.1], 
                                 forecast_quantiles[0.9],
                                 alpha=0.2, color='red')
    axes[0, 0].set_title('Forecast vs Actual', fontweight='bold')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Forecast Errors
    colors = ['red' if e > 0 else 'green' for e in errors]
    axes[0, 1].bar(range(1, forecast_period + 1), errors, color=colors)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_title('Daily Forecast Errors', fontweight='bold')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Error (Forecast - Actual)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Absolute Percentage Errors
    ape = np.abs(errors / actual_values) * 100
    mape = np.mean(ape)
    axes[1, 0].bar(range(1, forecast_period + 1), ape, color='steelblue', alpha=0.7)
    axes[1, 0].axhline(y=mape, color='red', linestyle='--', 
                        linewidth=2, label=f'Mean APE: {mape:.1f}%')
    axes[1, 0].set_title('Absolute Percentage Error by Day', fontweight='bold')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Absolute % Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Uncertainty Width
    if 0.1 in forecast_quantiles and 0.9 in forecast_quantiles:
        ci_width = forecast_quantiles[0.9] - forecast_quantiles[0.1]
        axes[1, 1].plot(range(1, forecast_period + 1), ci_width, 'o-', 
                        color='purple', linewidth=2, markersize=8)
        axes[1, 1].set_title('Forecast Uncertainty (80% CI Width)', fontweight='bold')
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('CI Width')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Quantiles not available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Uncertainty Analysis', fontweight='bold')
    
    plt.suptitle(f'{model_name} Error Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Error analysis saved as '{save_path}'")
    
    plt.show()


def compare_models(
    results: Dict[str, Dict[str, float]],
    save_path: str = None
) -> None:
    """
    Compare multiple models side by side.
    
    Args:
        results: Dict of {model_name: metrics_dict}
        save_path: Path to save plot (optional)
        
    Example:
        >>> results = {
        ...     'DeepAR': {'mae': 1000, 'rmse': 1500, 'mape': 12.5},
        ...     'SimpleFeedForward': {'mae': 1200, 'rmse': 1700, 'mape': 15.0}
        ... }
        >>> compare_models(results)
    """
    metrics_to_plot = ['mae', 'rmse', 'mape']
    model_names = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [results[model][metric] for model in model_names]
        axes[idx].bar(model_names, values, color=['steelblue', 'green', 'purple'][:len(model_names)])
        axes[idx].set_title(metric.upper(), fontweight='bold')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison saved as '{save_path}'")
    
    plt.show()
    
    # Print table
    print("\n📊 Model Comparison Table:")
    print("=" * 70)
    print(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
    print("-" * 70)
    for model, metrics in results.items():
        print(f"{model:<20} {metrics['mae']:>12,.2f} {metrics['rmse']:>12,.2f} {metrics['mape']:>11.2f}%")
    print("=" * 70)
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['mape'])
    print(f"\n🏆 Best Model (by MAPE): {best_model[0]} ({best_model[1]['mape']:.2f}%)")


# Quick reference
if __name__ == "__main__":
    print("=" * 70)
    print("Evaluation Utilities")
    print("=" * 70)
    print("\nAvailable functions:")
    print("\n1. calculate_metrics(forecast, actual)")
    print("   - Returns: {mae, rmse, mape, me, max_error}")
    print("\n2. print_metrics(metrics, model_name)")
    print("   - Pretty print metrics with interpretation")
    print("\n3. plot_forecast(...)")
    print("   - Comprehensive forecast visualization")
    print("\n4. plot_error_analysis(...)")
    print("   - Detailed error analysis plots")
    print("\n5. compare_models(results)")
    print("   - Compare multiple models side-by-side")
    print("=" * 70)

