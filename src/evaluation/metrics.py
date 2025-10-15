"""
Evaluation metrics for time series forecasting
"""

import numpy as np
import pandas as pd


def calculate_mae(actual, predicted):
    """Mean Absolute Error"""
    return np.mean(np.abs(actual - predicted))


def calculate_rmse(actual, predicted):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_mape(actual, predicted):
    """Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_smape(actual, predicted):
    """Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    # Avoid division by zero
    mask = denominator != 0
    return np.mean(numerator[mask] / denominator[mask]) * 100


def calculate_all_metrics(actual, predicted):
    """
    Calculate all standard metrics
    
    Returns:
        dict with MAE, RMSE, MAPE, sMAPE
    """
    return {
        'MAE': calculate_mae(actual, predicted),
        'RMSE': calculate_rmse(actual, predicted),
        'MAPE': calculate_mape(actual, predicted),
        'sMAPE': calculate_smape(actual, predicted)
    }


def evaluate_model(actual, predicted, model_name='Model'):
    """
    Evaluate a single model and print results
    
    Args:
        actual: Actual values
        predicted: Predicted values
        model_name: Name of the model
        
    Returns:
        Dictionary of metrics
    """
    metrics = calculate_all_metrics(actual, predicted)
    
    print(f"\n{model_name} Performance:")
    print(f"  MAE:   {metrics['MAE']:.2f}")
    print(f"  RMSE:  {metrics['RMSE']:.2f}")
    print(f"  MAPE:  {metrics['MAPE']:.2f}%")
    print(f"  sMAPE: {metrics['sMAPE']:.2f}%")
    
    return metrics


def compare_models(results_dict):
    """
    Compare multiple models
    
    Args:
        results_dict: Dict of model_name -> metrics_dict
        
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results_dict).T
    df.index.name = 'Model'
    df = df.reset_index()
    
    # Sort by RMSE (lower is better)
    df = df.sort_values('RMSE')
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(df.to_string(index=False))
    print("\nBest model (by RMSE):", df.iloc[0]['Model'])
    
    return df


def calculate_coverage(actual, lower, upper):
    """
    Calculate coverage of prediction intervals
    
    Args:
        actual: Actual values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        
    Returns:
        Coverage percentage (0-100)
    """
    in_interval = (actual >= lower) & (actual <= upper)
    coverage = np.mean(in_interval) * 100
    return coverage


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    actual = np.array([100, 105, 110, 115, 120])
    predicted = np.array([98, 107, 108, 118, 119])
    
    print("Example Evaluation:")
    metrics = evaluate_model(actual, predicted, model_name='Example Model')
    
    # Compare multiple models
    results = {
        'Model A': {'MAE': 10, 'RMSE': 12, 'MAPE': 8.5, 'sMAPE': 8.0},
        'Model B': {'MAE': 15, 'RMSE': 18, 'MAPE': 12.0, 'sMAPE': 11.5},
        'Model C': {'MAE': 8, 'RMSE': 10, 'MAPE': 7.0, 'sMAPE': 6.8}
    }
    
    comparison_df = compare_models(results)

