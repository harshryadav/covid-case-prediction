"""
Model Training Utilities for GluonTS COVID-19 Forecasting

This module provides convenient wrapper functions for training the three GluonTS models
we're using in this project: DeepAR, SimpleFeedForward, and DeepNPTS.

Each wrapper handles the model configuration and training, returning a predictor ready
to make forecasts. This keeps the notebook code clean and focuses on the story we're
telling rather than configuration details.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.torch.model.deep_npts import DeepNPTSEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator


@dataclass
class ModelResults:
    """
    Container for model training and evaluation results.
    
    This makes it easy to pass around everything we need: the trained model,
    its forecasts, and performance metrics.
    """
    model_name: str
    predictor: object
    forecasts: List
    ground_truths: List
    metrics: Dict
    training_time: float = 0.0


def train_deepar_covid(
    train_ds,
    test_ds,
    prediction_length: int = 14,
    num_feat_dynamic_real: int = 0,
    epochs: int = 20,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    num_layers: int = 2,
    hidden_size: int = 40,
    dropout: float = 0.1,
    verbose: bool = True
) -> ModelResults:
    """
    Train a DeepAR model on COVID-19 data.
    
    DeepAR is great for COVID forecasting because it uses recurrent neural networks
    to capture complex temporal patterns like the multiple pandemic waves and weekly
    reporting cycles we see in the data.
    
    Args:
        train_ds: GluonTS training dataset
        test_ds: GluonTS test dataset (with full history)
        prediction_length: How many days ahead to forecast (default: 14)
        num_feat_dynamic_real: Number of exogenous features (0 if none)
        epochs: Training iterations (default: 20)
        learning_rate: How fast the model learns (default: 0.001)
        context_length: Historical context to use (default: 2x prediction_length)
        num_layers: RNN depth - more layers = more complex patterns (default: 2)
        hidden_size: Network capacity (default: 40)
        dropout: Regularization to prevent overfitting (default: 0.1)
        verbose: Print training progress (default: True)
    
    Returns:
        ModelResults with trained predictor, forecasts, and metrics
    
    Example:
        >>> results = train_deepar_covid(train_ds, test_ds, num_feat_dynamic_real=3)
        >>> print(f"DeepAR MAPE: {results.metrics['MAPE']:.2f}%")
    """
    if verbose:
        print("\n" + "=" * 70)
        print("🚀 TRAINING DeepAR MODEL")
        print("=" * 70)
        print("\nDeepAR uses recurrent neural networks to learn temporal patterns.")
        print("Perfect for COVID data with multiple waves and weekly cycles!")
        print(f"\nConfiguration:")
        print(f"  • Epochs: {epochs}")
        print(f"  • Context length: {context_length or prediction_length * 2}")
        print(f"  • Features: {num_feat_dynamic_real}")
        print(f"  • Hidden size: {hidden_size} (network capacity)")
        print(f"  • Layers: {num_layers} (RNN depth)")
    
    import time
    start_time = time.time()
    
    # Configure the model
    estimator = DeepAREstimator(
        freq='D',
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout,
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={"max_epochs": epochs}
    )
    
    # Train the model
    if verbose:
        print("\n📚 Training in progress...")
        print("(This uses PyTorch Lightning under the hood)")
    
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Training complete in {training_time:.1f} seconds!")
    
    # Generate forecasts
    if verbose:
        print("\n🔮 Generating probabilistic forecasts...")
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100  # Generate 100 sample paths for uncertainty
    )
    
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    
    # Calculate metrics
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    
    if verbose:
        print(f"\n📊 DeepAR Performance:")
        print(f"  • MAPE: {agg_metrics.get('MAPE', 0):.2f}%")
        print(f"  • RMSE: {agg_metrics.get('RMSE', 0):.2f}")
        print(f"  • MAE: {agg_metrics.get('MAE', 0):.2f}")
        print("=" * 70)
    
    return ModelResults(
        model_name="DeepAR",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time
    )


def train_feedforward_covid(
    train_ds,
    test_ds,
    prediction_length: int = 14,
    epochs: int = 100,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    hidden_dimensions: Optional[List[int]] = None,
    verbose: bool = True
) -> ModelResults:
    """
    Train a SimpleFeedForward model on COVID-19 data.
    
    SimpleFeedForward is your baseline model - a simple neural network that's
    fast to train and works well when you have stable trends. Great for quick
    experiments or when you need fast retraining!
    
    NOTE: SimpleFeedForward doesn't support external features (num_feat_dynamic_real)
    or freq parameters. It's a simple model that only uses historical values.
    
    Args:
        train_ds: GluonTS training dataset
        test_ds: GluonTS test dataset (with full history)
        prediction_length: How many days ahead to forecast (default: 14)
        epochs: Training iterations (default: 100, trains fast!)
        learning_rate: How fast the model learns (default: 0.001)
        context_length: Historical context to use (default: 2x prediction_length)
        hidden_dimensions: Network architecture (default: [40, 40])
        verbose: Print training progress (default: True)
    
    Returns:
        ModelResults with trained predictor, forecasts, and metrics
    
    Example:
        >>> results = train_feedforward_covid(train_ds, test_ds)
        >>> print(f"SimpleFeedForward trained in {results.training_time:.1f}s!")
    """
    if verbose:
        print("\n" + "=" * 70)
        print("🚀 TRAINING SimpleFeedForward MODEL")
        print("=" * 70)
        print("\nSimpleFeedForward is a fast baseline using a simple neural network.")
        print("Perfect for stable trends and quick experiments!")
        print("\n⚠️  Note: This model doesn't use external features (deaths, mobility).")
        print("   It only learns from historical case patterns.")
        print(f"\nConfiguration:")
        print(f"  • Epochs: {epochs}")
        print(f"  • Context length: {context_length or prediction_length * 2}")
        print(f"  • Hidden layers: {hidden_dimensions or [40, 40]}")
    
    import time
    start_time = time.time()
    
    # Configure the model (no freq or num_feat_dynamic_real!)
    estimator = SimpleFeedForwardEstimator(
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        hidden_dimensions=hidden_dimensions or [40, 40],
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={"max_epochs": epochs}
    )
    
    # Train the model
    if verbose:
        print("\n📚 Training in progress...")
        print("(This should be fast - SimpleFeedForward trains quickly!)")
    
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Training complete in {training_time:.1f} seconds!")
        print("   (Told you it was fast! 😊)")
    
    # Generate forecasts
    if verbose:
        print("\n🔮 Generating probabilistic forecasts...")
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    
    # Calculate metrics
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    
    if verbose:
        print(f"\n📊 SimpleFeedForward Performance:")
        print(f"  • MAPE: {agg_metrics.get('MAPE', 0):.2f}%")
        print(f"  • RMSE: {agg_metrics.get('RMSE', 0):.2f}")
        print(f"  • MAE: {agg_metrics.get('MAE', 0):.2f}")
        print("=" * 70)
    
    return ModelResults(
        model_name="SimpleFeedForward",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time
    )


def train_deepnpts_covid(
    train_ds,
    test_ds,
    prediction_length: int = 14,
    num_feat_dynamic_real: int = 0,
    epochs: int = 30,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    num_hidden_nodes: Optional[List[int]] = None,
    dropout_rate: float = 0.1,
    verbose: bool = True
) -> ModelResults:
    """
    Train a DeepNPTS model on COVID-19 data.
    
    DeepNPTS (Deep Non-Parametric Time Series) is special - it doesn't assume
    your data follows a specific distribution. This makes it great for COVID data
    where the patterns can shift dramatically between waves!
    
    NOTE: DeepNPTS uniquely accepts 'epochs' as a direct parameter (not via trainer_kwargs)!
    
    Args:
        train_ds: GluonTS training dataset
        test_ds: GluonTS test dataset (with full history)
        prediction_length: How many days ahead to forecast (default: 14)
        num_feat_dynamic_real: Number of exogenous features (0 if none)
        epochs: Training iterations (default: 30) - DeepNPTS accepts this directly!
        learning_rate: How fast the model learns (default: 0.001)
        context_length: Historical context to use (default: 2x prediction_length)
        num_hidden_nodes: Network architecture (default: [40])
        dropout_rate: Regularization (default: 0.1)
        verbose: Print training progress (default: True)
    
    Returns:
        ModelResults with trained predictor, forecasts, and metrics
    
    Example:
        >>> results = train_deepnpts_covid(train_ds, test_ds, num_feat_dynamic_real=3)
        >>> print(f"DeepNPTS handles regime changes well!")
    """
    if verbose:
        print("\n" + "=" * 70)
        print("🚀 TRAINING DeepNPTS MODEL")
        print("=" * 70)
        print("\nDeepNPTS uses a non-parametric approach - it doesn't assume")
        print("your data follows any specific distribution.")
        print("Great for COVID data with shifting patterns!")
        print(f"\nConfiguration:")
        print(f"  • Epochs: {epochs} (passed directly to DeepNPTS!)")
        print(f"  • Context length: {context_length or prediction_length * 2}")
        print(f"  • Features: {num_feat_dynamic_real}")
        print(f"  • Hidden nodes: {num_hidden_nodes or [40]}")
        print(f"  • Dropout: {dropout_rate}")
    
    import time
    start_time = time.time()
    
    # Configure the model (DeepNPTS has epochs as direct parameter!)
    estimator = DeepNPTSEstimator(
        freq='D',
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_hidden_nodes=num_hidden_nodes or [40],
        dropout_rate=dropout_rate,
        epochs=epochs,  # DeepNPTS accepts epochs directly!
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50
    )
    
    # Train the model
    if verbose:
        print("\n📚 Training in progress...")
        print("(DeepNPTS learns flexible patterns from your data)")
    
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Training complete in {training_time:.1f} seconds!")
    
    # Generate forecasts
    if verbose:
        print("\n🔮 Generating probabilistic forecasts...")
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    
    # Calculate metrics
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    
    if verbose:
        print(f"\n📊 DeepNPTS Performance:")
        print(f"  • MAPE: {agg_metrics.get('MAPE', 0):.2f}%")
        print(f"  • RMSE: {agg_metrics.get('RMSE', 0):.2f}")
        print(f"  • MAE: {agg_metrics.get('MAE', 0):.2f}")
        print("=" * 70)
    
    return ModelResults(
        model_name="DeepNPTS",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time
    )


def compare_models(results_list: List[ModelResults]) -> pd.DataFrame:
    """
    Create a comparison table of multiple trained models.
    
    This makes it easy to see which model performed best on your COVID data.
    
    Args:
        results_list: List of ModelResults from training functions
    
    Returns:
        DataFrame with metrics for each model, sorted by MAPE (best first)
    
    Example:
        >>> deepar_results = train_deepar_covid(...)
        >>> feedforward_results = train_feedforward_covid(...)
        >>> deepnpts_results = train_deepnpts_covid(...)
        >>> 
        >>> comparison = compare_models([deepar_results, feedforward_results, deepnpts_results])
        >>> print(comparison)
    """
    comparison_data = []
    
    for results in results_list:
        comparison_data.append({
            'Model': results.model_name,
            'MAPE (%)': results.metrics.get('MAPE', np.nan),
            'RMSE': results.metrics.get('RMSE', np.nan),
            'MAE': results.metrics.get('MAE', np.nan),
            'Training Time (s)': results.training_time,
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by MAPE (lower is better)
    df = df.sort_values('MAPE (%)')
    
    # Add rank
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Pretty print the model comparison table.
    
    Args:
        comparison_df: DataFrame from compare_models()
    
    Example:
        >>> comparison = compare_models([...])
        >>> print_model_comparison(comparison)
    """
    print("\n" + "=" * 80)
    print("🏆 MODEL COMPARISON - COVID-19 FORECASTING")
    print("=" * 80)
    print("\nWhich model won? Let's see...\n")
    print(comparison_df.to_string(index=False))
    print("\n" + "=" * 80)
    
    # Declare the winner
    winner = comparison_df.iloc[0]
    print(f"\n🥇 Winner: {winner['Model']} with MAPE of {winner['MAPE (%)']:.2f}%")
    
    # Give context
    if winner['Model'] == 'DeepAR':
        print("\nDeepAR captured the complex COVID patterns well!")
    elif winner['Model'] == 'SimpleFeedForward':
        print("\nSimpleFeedForward proved simple can be powerful!")
    elif winner['Model'] == 'DeepNPTS':
        print("\nDeepNPTS handled the distribution shifts perfectly!")
    
    print("=" * 80 + "\n")


# Helpful utilities
def get_forecast_dataframe(
    forecast,
    ground_truth,
    start_date: pd.Timestamp,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Convert GluonTS forecast and ground truth to a convenient DataFrame.
    
    Makes it easier to work with forecasts for plotting and analysis.
    
    Args:
        forecast: GluonTS Forecast object
        ground_truth: Ground truth values
        start_date: When the forecast starts
        freq: Frequency (default: 'D' for daily)
    
    Returns:
        DataFrame with dates, predictions, actuals, and confidence intervals
    """
    forecast_length = len(forecast.mean)
    dates = pd.date_range(start=start_date, periods=forecast_length, freq=freq)
    
    df = pd.DataFrame({
        'Date': dates,
        'Prediction': forecast.mean,
        'Actual': ground_truth[-forecast_length:],
        'Lower_10': forecast.quantile(0.1),
        'Lower_25': forecast.quantile(0.25),
        'Median': forecast.quantile(0.5),
        'Upper_75': forecast.quantile(0.75),
        'Upper_90': forecast.quantile(0.9)
    })
    
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("GluonTS Model Training Utilities")
    print("=" * 70)
    print("\nThis module provides wrapper functions for training GluonTS models")
    print("on COVID-19 data. It makes your notebooks cleaner and more focused")
    print("on the story you're telling.")
    print("\nAvailable functions:")
    print("  • train_deepar_covid()")
    print("  • train_feedforward_covid()")
    print("  • train_deepnpts_covid()")
    print("  • compare_models()")
    print("  • print_model_comparison()")
    print("\nEach training function returns a ModelResults object with:")
    print("  - predictor (trained model)")
    print("  - forecasts (predictions)")
    print("  - ground_truths (actual values)")
    print("  - metrics (MAE, RMSE, MAPE, etc.)")
    print("  - training_time (seconds)")
    print("=" * 70)

